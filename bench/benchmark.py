"""推理基准：模型加载时间、单窗延迟 P50/P99、批量吞吐、显存峰值。

两侧统一口径（verifier=False，不含 CLIP 辅助验证——VLM 推理占绝对大头）：
  本地:   python bench/benchmark.py --model models/qwen2.5-vl-3b --adapter models/adapter_weighted --load4bit --tag int4-3060
  服务器: python bench/benchmark.py --model outputs/merged_weighted --tag bf16-5090
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load4bit", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=30, help="单窗延迟采样次数")
    args = ap.parse_args()

    import torch
    from engine.classifier import TileClassifier
    from engine.tiler import compute_tiles, crop_tiles

    # 测试图：来自验证集的不同类别，滑窗出一个 448 窗口
    imgs = []
    for cls in ["forest", "river", "harbor"]:
        files = sorted(Path(f"data/images/validation/{cls}").glob("*.jpg"))[:2]
        for f in files:
            imgs += crop_tiles(Image.open(f).convert("RGB"), compute_tiles(448, 448))[:1]
    while len(imgs) < args.n:
        imgs.append(imgs[len(imgs) % 3])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    clf = TileClassifier(args.model, adapter=args.adapter, load4bit=args.load4bit,
                         verifier=False)
    load_s = round(time.perf_counter() - t0, 2)

    for img in imgs[:3]:  # 预热
        clf.classify_tiles([img])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    lats = []
    for img in imgs[: args.n]:
        t = time.perf_counter()
        clf.classify_tiles([img])
        lats.append(time.perf_counter() - t)
    lats.sort()

    t = time.perf_counter()
    clf.classify_tiles(imgs[:16])
    batch16_s = time.perf_counter() - t

    result = {
        "tag": args.tag,
        "model": args.model,
        "quant": "nf4-int4" if args.load4bit else "bf16",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_load_s": load_s,
        "tile_latency_p50_ms": round(statistics.median(lats) * 1000, 1),
        "tile_latency_p99_ms": round(lats[int(len(lats) * 0.99) - 1] * 1000, 1),
        "throughput_tiles_per_s": round(len(imgs[:16]) / batch16_s, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None,
    }
    out = Path(f"eval/bench_{args.tag}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
