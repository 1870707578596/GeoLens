# GeoLens Phase 1 —— parquet 解包为 imagefolder
# 用法：
#   python scripts/05_extract_parquet.py --parquet-dir data/resisc45/data --out-dir data/images --manifest data/manifest.csv
# 说明：
#   timm/resisc45 是 parquet 格式（train/val/test 三个分片），此脚本解包为
#   data/images/{split}/{class}/{image_id}.jpg 的标准 imagefolder 结构，
#   并产出 manifest.csv（path, split, gt, image_id）供打标与后续评测使用。

import argparse
import csv
import io
from pathlib import Path

import pandas as pd
from PIL import Image

# 与 10_clip_label.py、HF README 中的 class_label 顺序保持一致
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


def save_image(cell, out_path: Path):
    """cell 为 HF image 列的取值：{'bytes': ..., 'path': ...}；异常时走 PIL 解码兜底。"""
    if isinstance(cell, dict) and cell.get("bytes"):
        out_path.write_bytes(cell["bytes"])
        return
    raw = cell.get("bytes") if isinstance(cell, dict) else None
    if raw is None and hasattr(cell, "read"):
        raw = cell.read()
    if raw:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(out_path.with_suffix(".jpg"), quality=95)
        return
    raise ValueError(f"无法解析图像单元格 -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    parquets = sorted(args.parquet_dir.glob("*.parquet"))
    assert parquets, f"no parquet found under {args.parquet_dir}"

    for pq in parquets:
        split = pq.name.split("-")[0]  # train-xxx.parquet / validation-xxx.parquet / test-xxx.parquet
        df = pd.read_parquet(pq)
        print(f"{pq.name}: {len(df)} rows, columns={list(df.columns)}", flush=True)
        for i, rec in enumerate(df.itertuples(index=False)):
            cls = RESISC45_CLASSES[int(rec.label)]
            image_id = getattr(rec, "image_id", None) or f"{cls}_{i:06d}"
            sub = args.out_dir / split / cls
            sub.mkdir(parents=True, exist_ok=True)
            out_path = sub / f"{Path(str(image_id)).stem}{Path(str(image_id)).suffix or '.jpg'}"
            save_image(rec.image, out_path)
            rows.append((str(out_path).replace("\\", "/"), split, cls, str(image_id)))
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(df)}", flush=True)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "gt", "image_id"])
        w.writerows(rows)
    print(f"done: {len(rows)} images -> {args.out_dir} | manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
