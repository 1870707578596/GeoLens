"""窗口分类器：微调 VLM 问答 + CLIP 交叉验证置信度。

- top1：weighted 微调的 Qwen2.5-VL 用中文问答给出类别（engine.vocab 解析）
- conf：CLIP ViT-B/16 零样本对 VLM 所选类别的 softmax 概率——
  两个模型不一致时 conf 天然偏低，低置信窗口即"建议人工复核"的候选
"""

from PIL import Image

from engine.vocab import CLASS_ZH, parse_class

QUESTION = "这张遥感影像中的主要地物是什么？请直接回答地物名称。"


class TileClassifier:
    def __init__(self, model_path: str, adapter: str | None = None,
                 load4bit: bool = False, batch_size: int = 8,
                 verifier: bool = True, device: str | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if load4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.clip_model = None
        if verifier:
            self._load_verifier()

    def _load_verifier(self):
        import open_clip

        clip, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai")
        self.clip_model = clip.to(self.device).eval()
        if self.device == "cuda":
            self.clip_model = self.clip_model.half()
        self.clip_preprocess = preprocess
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
        template = "a centered satellite photo of a {}."
        with self.torch.no_grad():
            texts = self.clip_tokenizer([template.format(c) for c in CLASS_ZH]).to(self.device)
            feats = self.clip_model.encode_text(texts)
            self.text_feats = feats / feats.norm(dim=-1, keepdim=True)

    def _vlm_classify(self, images: list[Image.Image]) -> list[str | None]:
        results = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            for img in batch:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": QUESTION},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
                with self.torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
                answer = self.processor.decode(
                    out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                results.append(parse_class(answer))
        return results

    def _clip_conf(self, images: list[Image.Image], classes: list[str | None]) -> list[float]:
        confs = []
        class_list = list(CLASS_ZH.keys())
        for img, cls in zip(images, classes):
            if cls is None or self.clip_model is None:
                confs.append(0.0 if cls is None else 1.0)
                continue
            x = self.clip_preprocess(img.convert("RGB"))
            if self.device == "cuda":
                x = x.half()
            with self.torch.no_grad():
                f = self.clip_model.encode_image(x.unsqueeze(0).to(self.device))
                f = f / f.norm(dim=-1, keepdim=True)
                logits = (f @ self.text_feats.T) * self.clip_model.logit_scale.exp()
                probs = logits.float().softmax(dim=-1)[0]
            confs.append(round(probs[class_list.index(cls)].item(), 4))
        return confs

    def classify_tiles(self, images: list[Image.Image]) -> list[dict]:
        """返回每窗 {top1, conf}；top1 为英文类名或 None（解析失败）。"""
        classes = self._vlm_classify(images)
        confs = self._clip_conf(images, classes)
        return [{"top1": c, "conf": p} for c, p in zip(classes, confs)]
