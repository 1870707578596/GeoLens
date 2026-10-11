from PIL import Image

from engine.exporter import build_report, render_overlay, to_geojson
from engine.vocab import CLASS_ZH

TILES = [
    {"x": 0, "y": 0, "w": 100, "h": 100, "top1": "forest", "conf": 0.9},
    {"x": 100, "y": 0, "w": 100, "h": 100, "top1": "river", "conf": 0.2},
]


def test_geojson_structure():
    gj = to_geojson(TILES)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 2
    f = gj["features"][0]
    assert f["geometry"]["type"] == "Polygon"
    assert f["geometry"]["coordinates"][0][0] == [0, 0]
    assert f["properties"]["class_en"] == "forest"
    assert f["properties"]["class_zh"] == CLASS_ZH["forest"][0]
    assert f["properties"]["conf"] == 0.9


def test_geojson_world_coords():
    aff = (116.0, 1e-4, 0.0, 40.0, 0.0, -1e-4)  # x0, dx, ry, y0, rx, dy
    gj = to_geojson(TILES, affine=aff)
    # 左上角像素(0,0) -> 世界(116.0, 40.0)；像素(100,0) -> (116.01, 40.0)
    assert gj["features"][0]["geometry"]["coordinates"][0][0] == [116.0, 40.0]
    assert abs(gj["features"][0]["geometry"]["coordinates"][0][1][0] - 116.01) < 1e-9


def test_overlay_size(tmp_path):
    img = Image.new("RGB", (200, 100))
    out = tmp_path / "overlay.png"
    render_overlay(img, TILES, out)
    assert out.exists()
    assert Image.open(out).size == (200, 100)


def test_report_content(tmp_path):
    stats = {
        "n_tiles": 2, "class_hist": {"forest": 1, "river": 1},
        "class_ratio": {"forest": 0.5, "river": 0.5}, "dominant": "forest",
        "low_conf_tiles": 1, "low_conf_ratio": 0.5, "low_conf_list": [TILES[1]],
    }
    out = tmp_path / "report.md"
    build_report(stats, out)
    text = out.read_text(encoding="utf-8")
    assert "森林" in text           # 众数类中文名
    assert "建议人工复核" in text    # 低置信提示
    assert "river" in text          # 类别表
