from engine.aggregator import aggregate

TILES = [
    {"x": 0, "y": 0, "w": 448, "h": 448, "top1": "forest", "conf": 0.9},
    {"x": 336, "y": 0, "w": 448, "h": 448, "top1": "forest", "conf": 0.8},
    {"x": 0, "y": 336, "w": 448, "h": 448, "top1": "river", "conf": 0.3},
]


def test_hist():
    assert aggregate(TILES)["class_hist"] == {"forest": 2, "river": 1}


def test_ratio():
    assert aggregate(TILES)["class_ratio"]["forest"] == 2 / 3


def test_dominant():
    assert aggregate(TILES)["dominant"] == "forest"


def test_low_conf():
    assert abs(aggregate(TILES)["low_conf_ratio"] - 1 / 3) < 1e-9
    assert aggregate(TILES)["low_conf_tiles"] == 1
