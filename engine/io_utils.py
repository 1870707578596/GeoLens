"""统一图像加载：GeoTIFF（含地理坐标）与普通图片。"""

from pathlib import Path

from PIL import Image

TIF_EXTS = {".tif", ".tiff"}


def load_image(path) -> tuple[Image.Image, tuple | None, str | None]:
    """返回 (PIL.Image, GDAL六参数仿射 or None, crs字符串 or None)。

    六参数: x_geo = a + b*col + c*row; y_geo = d + e*col + f*row
    """
    p = Path(path)
    if p.suffix.lower() in TIF_EXTS:
        import numpy as np
        import rasterio

        with rasterio.open(p) as src:
            arr = src.read()
            t = src.transform
            affine = (t.c, t.a, t.b, t.f, t.d, t.e)
            crs = str(src.crs) if src.crs else None
        if arr.shape[0] == 1:
            img = Image.fromarray(arr[0]).convert("RGB")
        else:
            img = Image.fromarray(np.transpose(arr[:3], (1, 2, 0))).convert("RGB")
        return img, affine, crs
    return Image.open(p).convert("RGB"), None, None
