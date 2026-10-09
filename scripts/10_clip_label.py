# GeoLens Phase 1 —— CLIP 零样本打标
# 用法（geolens 环境就绪后）：
#   python scripts/10_clip_label.py --image-dir data/resisc45 --output data/labels.jsonl
# 说明：
#   - 递归扫描 image-dir 下所有 jpg/png，父文件夹名作为 ground-truth 类别（RESISC45 目录结构）
#   - CLIP ViT-B/16 对 45 类做零样本分类，输出每图 top-5 概率、max_prob、熵
#   - 断点续跑：输出文件里已有的图片自动跳过

import argparse
import json
import math
from pathlib import Path

import torch
import open_clip
from PIL import Image

RESISC45_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
    "island", "lake", "meadow", "medium_residential", "mobile_home_park",
    "mountain", "overpass", "palace", "parking_lot", "railway",
    "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
    "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
    "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland",
]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def entropy_of(probs: list[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--template", default="a satellite photo of a {}.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：读已完成记录
    done = set()
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} images already labeled")

    # 收集图片
    images = sorted(
        p for p in args.image_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS and str(p).replace("\\", "/") not in done
    )
    print(f"to label: {len(images)} images | device: {args.device}")

    # 模型与文本特征
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(args.device).eval()
    if args.device == "cuda":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(args.model)

    with torch.no_grad():
        texts = tokenizer([args.template.format(c) for c in RESISC45_CLASSES]).to(args.device)
        text_feats = model.encode_text(texts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    top_labels = []
    with open(args.output, "a", encoding="utf-8") as out, torch.no_grad():
        for i in range(0, len(images), args.batch_size):
            batch_paths = images[i : i + args.batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(args.device)
            if args.device == "cuda":
                imgs = imgs.half()
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (feats @ text_feats.T) * model.logit_scale.exp()
            probs = logits.float().softmax(dim=-1).cpu()

            for p, prob in zip(batch_paths, probs):
                order = prob.argsort(descending=True)[:5]
                rec = {
                    "path": str(p).replace("\\", "/"),
                    "gt": p.parent.name,
                    "top1": RESISC45_CLASSES[order[0].item()],
                    "max_prob": round(prob[order[0]].item(), 6),
                    "entropy": round(entropy_of(prob.tolist()), 6),
                    "top5": [(RESISC45_CLASSES[j.item()], round(prob[j].item(), 6)) for j in order],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["top1"] == rec["gt"]:
                    top_labels.append(rec)
            out.flush()
            if (i // args.batch_size) % 20 == 0:
                labeled = len(done) + i + len(batch_paths)
                acc = len(top_labels) / max(len(top_labels) + (labeled - len(done) - (i + len(batch_paths))) + len(top_labels) - len(top_labels), 1)
                print(f"  {labeled}/{len(done) + len(images)} | running top1-acc so far: {len(top_labels)/(i + len(batch_paths)):.3f}", flush=True)

    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
