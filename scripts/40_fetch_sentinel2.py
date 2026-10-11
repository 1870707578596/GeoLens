"""抓取真实 Sentinel-2 真彩色窗口，保存为带地理坐标的本地 GeoTIFF 样例。

用法（需代理环境变量指向可达外网的本地代理）：
  export HTTPS_PROXY=http://127.0.0.1:7890
  python scripts/40_fetch_sentinel2.py --bbox 4.15 51.88 4.45 52.02 \
      --max-cloud 10 --window 4096 --out samples/rotterdam_harbor.tif

流程：Earth Search STAC（collections=sentinel-2-c1-l2a）按 bbox 搜索 ->
客户端过滤云量取最新 -> visual(TCI) 8bit 真彩色 COG -> rasterio 中心窗口
读取 -> 保留 CRS/transform 写本地 GeoTIFF。
"""

import argparse
from pathlib import Path

import requests
import rasterio
from rasterio.windows import Window

STAC = "https://earth-search.aws.element84.com/v1/search"


def find_scene(bbox, max_cloud):
    body = {
        "collections": ["sentinel-2-c1-l2a"],
        "bbox": list(bbox),
        "datetime": "2023-01-01T00:00:00Z/2026-10-11T00:00:00Z",
        "limit": 60,
    }
    r = requests.post(STAC, json=body, timeout=30)
    r.raise_for_status()
    feats = r.json().get("features", [])
    ok = [f for f in feats
          if f["properties"].get("eo:cloud_cover", 100) <= max_cloud
          and "visual" in f.get("assets", {})]
    if not ok:
        raise SystemExit(f"no scene under {max_cloud}% cloud (total {len(feats)})")
    ok.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
    return ok[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("minLon", "minLat", "maxLon", "maxLat"))
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=4096, help="窗口边长(像素, 10m)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    scene = find_scene(args.bbox, args.max_cloud)
    href = scene["assets"]["visual"]["href"]
    print(f"scene: {scene['id']} | cloud {scene['properties']['eo:cloud_cover']:.1f}%")

    with rasterio.open(href) as src:
        cx, cy = src.width // 2, src.height // 2
        half = args.window // 2
        win = Window(cx - half, cy - half, args.window, args.window)
        data = src.read(window=win)
        profile = {
            "driver": "GTiff", "dtype": "uint8", "count": src.count,
            "height": win.height, "width": win.width,
            "crs": src.crs, "transform": src.window_transform(win),
            "compress": "lzw",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(data)
    print(f"saved: {args.out} ({args.window}x{args.window}@10m, "
          f"crs={profile['crs']}) -> 约 {args.window * 10 / 1000:.1f} km 窗口")


if __name__ == "__main__":
    main()
