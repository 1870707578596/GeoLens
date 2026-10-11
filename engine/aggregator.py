"""窗口级预测聚合为全图统计。"""

from collections import Counter

DEFAULT_LOW_CONF = 0.6


def aggregate(tiles: list[dict], low_conf_threshold: float = DEFAULT_LOW_CONF) -> dict:
    """tiles: [{x,y,w,h,top1,conf}, ...] -> 全图统计。"""
    n = len(tiles)
    hist = Counter(t["top1"] for t in tiles)
    low_conf = [t for t in tiles if t.get("conf", 0.0) < low_conf_threshold]
    dominant, _ = hist.most_common(1)[0]
    return {
        "n_tiles": n,
        "class_hist": dict(hist),
        "class_ratio": {c: k / n for c, k in hist.items()},
        "dominant": dominant,
        "low_conf_tiles": len(low_conf),
        "low_conf_ratio": len(low_conf) / n,
        "low_conf_list": [
            {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"],
             "top1": t["top1"], "conf": t.get("conf", 0.0)}
            for t in low_conf
        ],
    }
