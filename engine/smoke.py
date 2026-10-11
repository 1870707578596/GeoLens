"""冒烟验证：2 张已知类别图走完 切片→分类→聚合 全链路。

用法（模型与 adapter 就位后）：
  python engine/smoke.py --model models/qwen2.5-vl-3b --adapter models/adapter_weighted
"""

import argparse
import glob
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.aggregator import aggregate
from engine.classifier import TileClassifier
from engine.tiler import compute_tiles, crop_tiles

CASES = {"mountain": "data/images/test/mountain", "harbor": "data/images/test/harbor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load4bit", action="store_true")
    args = ap.parse_args()

    clf = TileClassifier(args.model, adapter=args.adapter, load4bit=args.load4bit)
    ok = 0
    for gt, folder in CASES.items():
        img_path = sorted(glob.glob(f"{folder}/*.jpg"))[0]
        img = Image.open(img_path).convert("RGB")
        tiles = compute_tiles(*img.size)
        preds = clf.classify_tiles(crop_tiles(img, tiles))
        stats = aggregate([dict(zip(("x", "y", "w", "h"), t), **p)
                           for t, p in zip(tiles, preds)])
        hit = stats["dominant"] == gt
        ok += hit
        print(f"{gt:>10}: dominant={stats['dominant']:<10} conf={preds[0]['conf']} {'OK' if hit else 'MISS'}")
    print("SMOKE", "PASS" if ok == len(CASES) else "FAIL")
    sys.exit(0 if ok == len(CASES) else 1)


if __name__ == "__main__":
    main()
