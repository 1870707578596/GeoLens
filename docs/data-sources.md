# 真实卫星数据来源

GeoLens Studio 支持任意 GeoTIFF（带 CRS/仿射坐标时输出真实经纬度的 GeoJSON）。

## 内置样例：Sentinel-2 真彩色（免注册、免 API key）

数据源：[Earth Search STAC](https://earth-search.aws.element84.com)（AWS 公开托管，
collection `sentinel-2-c1-l2a`）的 **TCI 真彩色 8bit COG**，通过 rasterio 窗口读取，
只传输感兴趣区域，无需下载整景。

```bash
# 需要能访问 aws 的网络环境（如本地代理）
export HTTPS_PROXY=http://127.0.0.1:7890
python scripts/40_fetch_sentinel2.py --bbox 4.15 51.88 4.45 52.02 \
    --max-cloud 10 --window 4096 --out samples/rotterdam_harbor.tif
```

当前样例：`S2A_T31UET_20260812`（2026-08-12，云量 0.4%），鹿特丹港 41km 窗口，
EPSG:32631。文件 38MB 不入 git，按上面命令随时重建。

## 其他免费获取途径

- [Copernicus Data Space](https://dataspace.copernicus.eu/)：官方完整 L1C/L2A（需注册账号）
- [AWS Open Data Sentinel-2](https://registry.opendata.aws/sentinel-2/)：本脚本所用公开 COG 池
- USGS EarthExplorer（Landsat 系，需注册）

## 关于坐标

TCI 的 10m 分辨率窗口用 UTM 投影（如 EPSG:32631）；导出的 GeoJSON 顶点为
UTM 坐标（与影像 CRS 一致）。如需 WGS84 经纬度，用 QGIS 打开后重投影即可，
或在使用端做 proj4 变换。
