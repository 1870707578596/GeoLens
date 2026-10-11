"""GeoLens Studio FastAPI 服务层。

启动（本地 int4）：
  GEOLENS_MODEL=models/qwen2.5-vl-3b GEOLENS_ADAPTER=models/adapter_weighted \
  GEOLENS_4BIT=1 uvicorn service.api:app --port 8000
测试时向 app.state.classifier 注入桩对象即可（无需真实模型）。
"""

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

TASKS_DIR = Path("outputs/tasks")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_BYTES = 500 * 1024 * 1024

app = FastAPI(title="GeoLens Studio API", version="2.0")
app.state.classifier = None
app.state.executor = None
app.state.tasks = {}
app.state.lock = threading.Lock()


def get_classifier():
    """惰性加载真实模型；测试里直接给 app.state.classifier 赋桩。"""
    if app.state.classifier is None:
        from engine.classifier import TileClassifier

        app.state.classifier = TileClassifier(
            model_path=os.environ.get("GEOLENS_MODEL", "models/qwen2.5-vl-3b"),
            adapter=os.environ.get("GEOLENS_ADAPTER") or None,
            load4bit=os.environ.get("GEOLENS_4BIT", "0") == "1",
        )
    return app.state.classifier


def run_task(task_id: str, path: Path):
    try:
        from engine.aggregator import aggregate
        from engine.exporter import build_report, render_overlay, to_geojson
        from engine.io_utils import load_image
        from engine.tiler import compute_tiles, crop_tiles

        img, affine, _crs = load_image(path)
        tiles = compute_tiles(*img.size)
        preds = get_classifier().classify_tiles(crop_tiles(img, tiles))
        tiles_full = [dict(zip(("x", "y", "w", "h"), t), **p) for t, p in zip(tiles, preds)]
        stats = aggregate(tiles_full)

        out = TASKS_DIR / task_id
        out.mkdir(parents=True, exist_ok=True)
        render_overlay(img, tiles_full, out / "overlay.png")
        (out / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "polygons.geojson").write_text(
            json.dumps(to_geojson(tiles_full, affine), ensure_ascii=False), encoding="utf-8")
        build_report(stats, out / "report.md", image_name=path.name)
        _update(task_id, status="done",
                files=["overlay.png", "stats.json", "polygons.geojson", "report.md"])
    except Exception as e:  # noqa: BLE001 —— 失败原因要回传给客户端
        _update(task_id, status="failed", error=f"{type(e).__name__}: {e}")


def _update(task_id: str, **kv):
    with app.state.lock:
        app.state.tasks.setdefault(task_id, {}).update(kv)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(415, f"不支持的格式 {suffix!r}，允许: {sorted(ALLOWED_EXTS)}")
    task_id = uuid.uuid4().hex[:12]
    dest = TASKS_DIR / task_id / f"input{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                raise HTTPException(413, "文件超过 500MB 上限")
            f.write(chunk)
    _update(task_id, status="queued")
    if app.state.executor is None:
        app.state.executor = ThreadPoolExecutor(max_workers=1)
    app.state.executor.submit(run_task, task_id, dest)
    return {"task_id": task_id}


@app.get("/result/{task_id}")
def result(task_id: str):
    with app.state.lock:
        task = dict(app.state.tasks.get(task_id, {}))
    if not task:
        raise HTTPException(404, "task not found")
    return {"task_id": task_id, **task}


@app.get("/result/{task_id}/download/{filename}")
def download(task_id: str, filename: str):
    path = (TASKS_DIR / task_id / filename).resolve()
    if not str(path).startswith(str(TASKS_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path)


@app.get("/health")
def health():
    import torch

    return {
        "status": "ok",
        "model_loaded": app.state.classifier is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
