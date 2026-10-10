# GeoLens Phase 2 —— zero-shot 基线评测（服务器，geolens 环境）
# 用法：
#   python eval/30_zeroshot_eval.py --n 450 --out eval/results_zeroshot.json
# 逻辑：
#   从 test split 分层抽样（45 类 × 每类 n/45 张），问"主要地物是什么"，
#   在生成的中文回答里匹配 45 类中文名，与目录名（gt）比对算精度。
#   该脚本同样适用于评测微调后的 LoRA 模型（--adapter 参数）。

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import importlib.util
_bpath = Path(__file__).resolve().parent.parent / "scripts" / "20_build_instructions.py"
_spec = importlib.util.spec_from_file_location("builder", _bpath)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)
CLASS_ZH = _builder.CLASS_ZH  # {en: (zh, desc)}
ZH2EN = {v[0]: k for k, v in CLASS_ZH.items()}
# 常见同义词归一（模型回答里的说法 → 标准类名）；有歧义的（如"农田""住宅区"）不映射
SYNONYMS = {
    "飞机场": "机场", "高速路": "高速公路", "铁轨": "铁路", "码头": "港口",
    "桥": "桥梁", "山地": "山脉", "云层": "云", "灌木": "灌木丛", "沙滩": "海滩",
    "轮船": "船舶", "船只": "船舶", "储罐": "储油罐", "油罐": "储油罐",
    "发电厂": "火力发电站", "电厂": "火力发电站", "跑道": "飞机跑道",
}

QUESTION = "这张遥感影像中的主要地物是什么？请直接回答地物名称。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录（评测微调模型时传入）")
    ap.add_argument("--image-root", default="data/images/test")
    ap.add_argument("--n", type=int, default=450, help="抽样总数（尽量 45 的倍数）")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_class = max(1, args.n // len(CLASS_ZH))
    root = Path(args.image_root)
    by_class = defaultdict(list)
    for p in root.rglob("*.jpg"):
        by_class[p.parent.name].append(p)
    samples = []
    for cls, files in sorted(by_class.items()):
        samples += [(f, cls) for f in rng.sample(files, min(per_class, len(files)))]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    zh_pattern = "|".join(sorted(ZH2EN.keys(), key=len, reverse=True))
    correct, unparsed = 0, 0
    records = []
    for i, (img_path, gt) in enumerate(samples):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": QUESTION},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[Image.open(img_path).convert("RGB")], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        answer = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        hits = re.findall(zh_pattern, answer)
        if hits:
            pred = ZH2EN[hits[0]]
        else:
            pred = None
            for s in sorted(SYNONYMS, key=len, reverse=True):
                if s in answer:
                    pred = ZH2EN[SYNONYMS[s]]
                    break
        if pred is None:
            unparsed += 1
        correct += pred == gt
        records.append({"image": str(img_path), "gt": gt, "pred": pred, "answer": answer.strip()})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} acc={correct/(i+1):.4f} unparsed={unparsed}", flush=True)

    acc = correct / len(samples)
    result = {
        "tag": args.tag, "model": args.model, "adapter": args.adapter,
        "n": len(samples), "accuracy": round(acc, 4), "unparsed_rate": round(unparsed / len(samples), 4),
    }
    args.out.write_text(json.dumps({"summary": result, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
