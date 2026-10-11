"""API 集成测试：注入桩分类器，走完 上传→任务→产物→下载 全流程。"""

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service import api


class StubClassifier:
    def classify_tiles(self, images):
        return [{"top1": "forest", "conf": 0.9} for _ in images]


@pytest.fixture()
def client():
    api.app.state.classifier = StubClassifier()  # 注入桩，跳过真实模型
    api.app.state.executor = None
    api.app.state.tasks.clear()
    with TestClient(api.app) as c:
        yield c
    api.app.state.classifier = None


def _png_bytes(size=(900, 900)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 80, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_full_flow(client):
    r = client.post("/analyze", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    for _ in range(50):
        r = client.get(f"/result/{task_id}")
        if r.json().get("status") in ("done", "failed"):
            break
        time.sleep(0.2)
    assert r.json()["status"] == "done", r.json()
    assert "overlay.png" in r.json()["files"]

    d = client.get(f"/result/{task_id}/download/report.md")
    assert d.status_code == 200
    assert "森林".encode() in d.content  # 桩全部预测 forest


def test_reject_bad_format(client):
    r = client.post("/analyze", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_download_traversal_blocked(client):
    r = client.get("/result/xxx/download/..%2F..%2FREADME.md")
    assert r.status_code == 404


def test_unknown_task(client):
    assert client.get("/result/nope").status_code == 404
