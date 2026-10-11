"""产物导出：分类叠加图、GeoJSON、Markdown 分析报告。"""

import colorsys
import datetime
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from engine.vocab import CLASS_ZH

LOW_CONF = 0.6


def class_color(en: str) -> tuple[int, int, int]:
    """按类别索引生成稳定色（黄金角散列）。"""
    idx = list(CLASS_ZH).index(en) if en in CLASS_ZH else 0
    r, g, b = colorsys.hsv_to_rgb((idx * 137.508) % 360 / 360, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int = 14):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlay(img: Image.Image, tiles: list[dict], out_path) -> None:
    """半透明分类色块 + 类别中文标签；低置信窗口红框标注。"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _font(max(14, img.width // 60))
    for t in tiles:
        if t["top1"] is None:
            continue
        color = class_color(t["top1"])
        box = (t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"])
        d.rectangle(box, fill=color + (80,), outline=color + (200,), width=2)
        if t.get("conf", 1.0) < LOW_CONF:
            d.rectangle(box, outline=(255, 40, 40, 255), width=5)
        d.text((t["x"] + 6, t["y"] + 4), CLASS_ZH[t["top1"]][0],
               fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(out_path)


def to_geojson(tiles: list[dict], affine: tuple | None = None) -> dict:
    """tiles -> FeatureCollection。affine=(x0,dx,ry,y0,rx,dy) 为 GDAL 风容
    的六参数，提供时输出真实地理坐标（GeoJSON 经纬度序），否则像素坐标。"""
    def to_world(px: int, py: int) -> list[float]:
        if affine is None:
            return [float(px), float(py)]
        x0, dx, ry, y0, rx, dy = affine
        return [x0 + dx * px + ry * py, y0 + rx * px + dy * py]

    features = []
    for t in tiles:
        if t["top1"] is None:
            continue
        ring = [to_world(t["x"], t["y"]), to_world(t["x"] + t["w"], t["y"]),
                to_world(t["x"] + t["w"], t["y"] + t["h"]), to_world(t["x"], t["y"] + t["h"]),
                to_world(t["x"], t["y"])]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "class_en": t["top1"],
                "class_zh": CLASS_ZH[t["top1"]][0],
                "conf": t.get("conf", 0.0),
                "low_conf": bool(t.get("conf", 0.0) < LOW_CONF),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def build_report(stats: dict, out_path, image_name: str = "") -> None:
    dominant_zh = CLASS_ZH[stats["dominant"]][0] if stats["dominant"] in CLASS_ZH else "未识别"
    lines = [
        "# 遥感影像分析报告",
        "",
        f"- 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- 影像：{image_name or '(未命名)'}",
        f"- 分析窗口数：{stats['n_tiles']}",
        f"- 主要地物（众数类）：{dominant_zh} ({stats['dominant']})",
        f"- 低置信窗口：{stats['low_conf_tiles']} 个（占比 {stats['low_conf_ratio']:.1%}）",
        "",
        "## 类别统计",
        "",
        "| 类别 | 英文 | 窗口数 | 占比 |",
        "|---|---|---|---|",
    ]
    for cls, n in sorted(stats["class_hist"].items(), key=lambda kv: -kv[1]):
        zh = CLASS_ZH.get(cls, ("未识别", ""))[0]
        lines.append(f"| {zh} | {cls} | {n} | {stats['class_ratio'][cls]:.1%} |")
    lines += ["", "## 低置信窗口（建议人工复核）", ""]
    if stats["low_conf_list"]:
        lines.append("| 位置 (x, y, w, h) | 类别 | 置信度 |")
        lines.append("|---|---|---|")
        for t in stats["low_conf_list"][:20]:
            zh = CLASS_ZH.get(t["top1"], ("未识别", ""))[0] if t["top1"] else "未识别"
            lines.append(f"| ({t['x']}, {t['y']}, {t['w']}, {t['h']}) | {zh} | {t['conf']:.2f} |")
        if len(stats["low_conf_list"]) > 20:
            lines.append(f"| ... 其余 {len(stats['low_conf_list']) - 20} 个略 | | |")
    else:
        lines.append("无——所有窗口置信度均达标。")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
