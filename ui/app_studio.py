"""GeoLens Studio —— Gradio 工具界面（只通过 HTTP 调用 FastAPI 服务）。

启动（先起 service.api）：
  python ui/app_studio.py --api http://127.0.0.1:8000 --port 7862
"""

import argparse
import tempfile
import time
from pathlib import Path

import gradio as gr
import httpx

TEMP = Path(tempfile.gettempdir()) / "geolens_studio"


def analyze(image, api_base, progress=gr.Progress()):
    """上传 → 轮询任务 → 拉取四件产物，返回展示内容。"""
    if image is None:
        raise gr.Error("请先上传遥感影像（JPG/PNG/GeoTIFF）")
    TEMP.mkdir(exist_ok=True)
    from PIL import Image

    p = TEMP / "input.png"
    Image.fromarray(image).save(p)

    with httpx.Client(timeout=30) as c:
        health = c.get(f"{api_base}/health").json()
        progress(0.05, desc="上传影像")
        with open(p, "rb") as f:
            r = c.post(f"{api_base}/analyze", files={"file": (p.name, f, "image/png")})
        r.raise_for_status()
        task_id = r.json()["task_id"]

        deadline = time.time() + 1800
        status = "queued"
        while time.time() < deadline:
            st = c.get(f"{api_base}/result/{task_id}").json()
            status = st.get("status")
            if status in ("done", "failed"):
                break
            time.sleep(3)
        if status != "done":
            raise gr.Error(f"分析失败: {st.get('error', status)}")

        files = {}
        for name in st["files"]:
            r = c.get(f"{api_base}/result/{task_id}/download/{name}")
            dest = TEMP / name
            dest.write_bytes(r.content)
            files[name] = dest

    import json

    stats = json.loads(files["stats.json"].read_text(encoding="utf-8"))
    from engine.vocab import CLASS_ZH

    hist = sorted(stats["class_hist"].items(), key=lambda kv: -kv[1])[:12]
    bar_data = [(CLASS_ZH.get(c, ("未知",))[0], n) for c, n in hist]
    report = files["report.md"].read_text(encoding="utf-8")
    info = (f"✅ 完成：{stats['n_tiles']} 个窗口 | 主要地物 "
            f"{CLASS_ZH.get(stats['dominant'], ('未知',))[0]} | "
            f"低置信 {stats['low_conf_tiles']} 个 ({stats['low_conf_ratio']:.0%})\n\n"
            f"推理设备：{health.get('gpu') or health.get('device')}")
    return (
        str(files["overlay.png"]),
        bar_data,
        report,
        [str(files["polygons.geojson"]), str(files["report.md"]), str(files["stats.json"])],
        info,
    )


def build(api_base: str):
    with gr.Blocks(title="GeoLens Studio") as app:
        gr.Markdown("# 🛰️ GeoLens Studio\n"
                    "上传大幅遥感影像（JPG / PNG / GeoTIFF），自动滑窗识别地物，"
                    "生成叠加图、统计、GeoJSON 与分析报告。低置信窗口已用红框标出，"
                    "建议人工复核。")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="遥感影像")
                api_box = gr.Textbox(value=api_base, label="API 地址", visible=True)
                btn = gr.Button("开始分析", variant="primary")
            with gr.Column():
                status = gr.Markdown()
                overlay = gr.Image(type="filepath", label="分类叠加图（红框=低置信）")
        bar = gr.BarPlot(x="地物", y="窗口数", title="地物分布 Top12",
                         vertical=False, height=260)
        with gr.Row():
            report_md = gr.Markdown(label="分析报告")
            downloads = gr.File(label="下载产物", file_count="multiple")
        btn.click(analyze, [img_in, api_box],
                  [overlay, bar, report_md, downloads, status])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build(args.api).queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
