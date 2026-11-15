# MOT17 detection + tracking + evaluation pipeline (CPU).
# Mount the dataset; do not bake MOT17 into the image.
#
#   docker build -t mot17-pipeline .
#   docker run --rm \
#     -v "$PWD/MOT17:/app/MOT17" \
#     -v "$PWD/mot_pipeline (Object_Detection + Tracking)/results:/app/mot_pipeline/results" \
#     -v "$PWD/mot_pipeline (Object_Detection + Tracking)/outputs:/app/mot_pipeline/outputs" \
#     mot17-pipeline
#
# Override the command to run a single stage, for example:
#   docker run --rm -v "$PWD/MOT17:/app/MOT17" mot17-pipeline \
#     detect.py --detector yolo26n --sequence MOT17-02-DPM --imgsz 1280

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY docker-compose.yml /app/docker-compose.yml
COPY ["mot_pipeline (Object_Detection + Tracking)/", "/app/mot_pipeline/"]

WORKDIR /app/mot_pipeline

ENTRYPOINT ["python"]
CMD ["run_benchmark.py", "--device", "cpu"]
