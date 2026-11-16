# Shared serving image. Each stage Deployment overrides the command and MODEL_NAME.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git g++ cmake libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt ./
RUN pip install --upgrade pip \
    && pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install ultralytics==8.4.214 opencv-python-headless "numpy>=1.24.0" pillow \
         lap trackeval minio onnx onnxruntime \
    && pip install -r requirements-serve.txt

COPY ["mot_pipeline (Object_Detection + Tracking)/", "/app/mot_pipeline (Object_Detection + Tracking)/"]
COPY services /app/services

EXPOSE 50051 8080
CMD ["python", "-m", "services.detect.server"]
