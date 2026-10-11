"""滑窗切片：把大幅影像切成模型可处理的窗口。

策略：全部窗口取完整 window 尺寸，最后一个窗口右/下边缘钳制到图像边界
（平移而非裁小），保证覆盖完整且无碎窗；图像小于 window 时不切分。
"""

from PIL import Image

MIN_SIZE = 64


def _axis_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    last = length - window
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def compute_tiles(w: int, h: int, window: int = 448, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """返回 (x, y, ww, hh) 窗口列表；图像小于 window 时返回整图单窗。"""
    if w < MIN_SIZE or h < MIN_SIZE:
        raise ValueError(f"image too small to tile: {w}x{h}")
    stride = int(window * (1 - overlap))
    tiles = []
    for x in _axis_positions(w, window, stride):
        for y in _axis_positions(h, window, stride):
            ww = min(window, w)
            hh = min(window, h)
            tiles.append((x, y, ww, hh))
    return tiles


def crop_tiles(img: Image.Image, tiles: list[tuple[int, int, int, int]]) -> list[Image.Image]:
    return [img.crop((x, y, x + ww, y + hh)) for x, y, ww, hh in tiles]
