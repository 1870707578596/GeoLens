from PIL import Image

from engine.tiler import compute_tiles, crop_tiles


def test_exact_fit():
    assert compute_tiles(448, 448) == [(0, 0, 448, 448)]


def test_cover_edges():
    ts = compute_tiles(1000, 700)
    assert ts[-1][0] + ts[-1][2] == 1000
    assert ts[-1][1] + ts[-1][3] == 700


def test_at_least_2x2():
    ts = compute_tiles(900, 900)
    assert len(ts) >= 4
    xs = sorted({t[0] for t in ts})
    assert xs[1] - xs[0] == 336  # stride = 448 * 0.75


def test_small_image_not_tiled():
    assert compute_tiles(300, 200) == [(0, 0, 300, 200)]


def test_crop_tiles():
    img = Image.new("RGB", (1000, 700))
    tiles = compute_tiles(1000, 700)
    crops = crop_tiles(img, tiles)
    assert len(crops) == len(tiles)
    assert crops[-1].size == (tiles[-1][2], tiles[-1][3])
