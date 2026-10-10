# GeoLens Phase 5 —— Gradio Demo：遥感影像中文问答
# 服务器运行（bf16 合并模型）:
#   python demo/app.py --model outputs/merged_weighted --share
# 本地运行（3060 6G，int4）:
#   python demo/app.py --model <本地合并模型路径> --load4bit
#   （或传 --model Qwen/Qwen2.5-VL-3B-Instruct --adapter <adapter路径> --load4bit）

import argparse

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_Q = "这张遥感影像中的主要地物是什么？"


def build_model(args):
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.load4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="合并后模型目录或 HF id")
    ap.add_argument("--adapter", default=None, help="可选：LoRA adapter（未合并时）")
    ap.add_argument("--load4bit", action="store_true", help="本地低显存用 int4 量化")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成 gradio 公网分享链接")
    args = ap.parse_args()

    model = build_model(args)
    processor = AutoProcessor.from_pretrained(args.model)

    def predict(image, question):
        if image is None:
            return "请先上传一张遥感影像。"
        if not question.strip():
            question = DEFAULT_Q
        pil = Image.fromarray(image).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    examples = [
        ["demo/examples/harbor.jpg", DEFAULT_Q],
        ["demo/examples/mountain.jpg", DEFAULT_Q],
        ["demo/examples/circular_farmland.jpg", DEFAULT_Q],
        ["demo/examples/airplane.jpg", "这张遥感影像中是否包含飞机？"],
    ]

    with gr.Blocks(title="GeoLens 遥感多模态助手") as app:
        gr.Markdown("# GeoLens 遥感多模态助手\n"
                    "Qwen2.5-VL-3B + 自动构建遥感指令数据 LoRA 微调（weighted 软加权，45 类地物 62.0%）。"
                    "上传遥感影像，用中文提问。")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="遥感影像")
                q_in = gr.Textbox(value=DEFAULT_Q, label="问题")
                btn = gr.Button("识别", variant="primary")
            out = gr.Textbox(label="回答", lines=6)
        btn.click(predict, [img_in, q_in], out)
        gr.Examples(examples, [img_in, q_in], label="示例（RESISC45 测试集）")

    app.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
