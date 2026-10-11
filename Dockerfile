# GeoLens Studio 服务镜像（int4 推理，适配 6GB 级 GPU）
# 注意：基础镜像标签若拉取失败，可换 nvidia/cuda:12.6.2-runtime-ubuntu22.04 + pip torch cu126
FROM pytorch/pytorch:2.7.1-cuda12.6-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir bitsandbytes fastapi uvicorn

COPY engine/ service/ ui/ scripts/ bench/ eval/ docs/ README.md LICENSE ./

# 模型挂载在 /models（docker-compose 已配置），int4 加载
ENV GEOLENS_MODEL=/models/qwen2.5-vl-3b \
    GEOLENS_ADAPTER=/models/adapter_weighted \
    GEOLENS_4BIT=1 \
    HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
