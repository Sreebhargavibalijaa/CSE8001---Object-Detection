# MOT17 Detection + Tracking + Evaluation Pipeline

Compares pretrained person detectors with two trackers on MOT17 and scores each
combination with the official TrackEval metrics plus a motion assertion.

```text
MOT17 images -> detect.py -> detections CSV -> track.py -> MOT track file -> evaluate.py -> metrics
```

| Stage | Options |
|---|---|
| Detector | YOLO26n (`imgsz` 640 / 960 / 1280), Faster R-CNN MobileNetV3-Large 320-FPN |
| Tracker | ByteTrack (default and low-threshold), OC-SORT |
| Metrics | HOTA, MOTA, IDF1, Precision, Recall, motion-assertion pass rate |

Everything uses pretrained weights. No training or fine-tuning is involved.

The seven MOT17 training videos are each shipped three times (`-DPM`, `-FRCNN`, `-SDP`)
with **identical frames**. Configuration choice uses **one copy of each video**
(`MOT17-02-DPM` … `MOT17-13-DPM`).

---

## Faster R-CNN

The original torchvision **ResNet-50 FPN** Faster R-CNN runs at about **0.13 FPS on CPU**.
A 7-sequence sweep is about 5,300 frames; that model would take around 11 hours per pass
and makes `imgsz` / tracker-threshold profiling impractical.

It is replaced by **Faster R-CNN MobileNetV3-Large 320-FPN**
(`fasterrcnn_mobilenet_v3_large_320_fpn`). It is still a two-stage Faster R-CNN, with a
mobile backbone and a 320 px input. On this machine it runs at **16 FPS on CPU**
(Apple MPS is slower for this torchvision model, so `detect.py` forces CPU). That is
fast enough to profile all seven videos in minutes. DETR ResNet-50 remains available
in `detect.py --detector detr` but is not part of the default sweep.

---

## Why YOLO26n recall is low at `imgsz=640`

YOLO26n at the default 640 px input is a small COCO model. MOT17 pedestrians are often
tens of pixels tall in 1920×1080 frames. After letterbox resize to 640, those people
occupy only a few pixels and are dropped. The 30-frame MOT17-02 check was already in
that regime: about **23% TrackEval recall at ~93% precision**.

Natural variants, both of which this repo profiles:

- **Larger `imgsz` (960, 1280)** so small people survive the resize
- **Lower ByteTrack thresholds** (`configs/bytetrack_low.yaml`) so low-score boxes
  from small people are allowed to start and continue tracks

`run_benchmark.py` also reports **detection** recall (IoU 0.5 vs GT pedestrians)
binned by box height, which separates “the detector never saw the person” from
“the tracker threw the box away”.

---

## Files

| File | What it does | Output |
|---|---|---|
| `detect.py` | Runs one detector over a sequence | `outputs/detections/<sequence>/<detector-tag>.csv` |
| `track.py` | ByteTrack or OC-SORT over saved boxes | `outputs/tracks/<sequence>/<detector>_<tracker>.txt` |
| `evaluate.py` | TrackEval metrics **and** motion assertion pass rate | `outputs/metrics/<sequence>/<detector>_<tracker>.csv` |
| `run_benchmark.py` | All 7 unique train sequences, YOLO `imgsz` and tracker variants | `results/summary.csv`, `results/REPORT.md` |
| `common.py` | Shared helpers | |
| `configs/` | ByteTrack default, ByteTrack low-threshold, OC-SORT | |

Each detector runs once per sequence; every tracker variant reuses those boxes.

---

## Setup

```bash
pip install -r requirements.txt
```

Place MOT17 at the repository root:

```text
CSE8001---Object-Detection/
├── MOT17/train/MOT17-02-DPM/   (img1/, gt/, seqinfo.ini)
├── mot_pipeline (Object_Detection + Tracking)/
└── Dockerfile
```

If the dataset is elsewhere, pass `--dataset-root`.

### Docker

```bash
docker build -t mot17-pipeline .
docker run --rm \
  -v "$PWD/MOT17:/app/MOT17" \
  -v "$PWD/mot_pipeline (Object_Detection + Tracking)/results:/app/mot_pipeline/results" \
  -v "$PWD/mot_pipeline (Object_Detection + Tracking)/outputs:/app/mot_pipeline/outputs" \
  mot17-pipeline
```

The image does not contain MOT17. The default command is the 7-sequence sweep on CPU.
YOLO weights are downloaded on first run.

---

## T2. Model variants and registry

Detection and tracking are the scalable stages. Each has **at least four** registered
variants (family / input knobs / ONNX / INT8 for detection; threshold and buffer
YAMLs for tracking). TensorRT is not used (CPU-only / no NVIDIA GPU); graph
optimization is **ONNX Runtime**, quantization is **ONNX Runtime dynamic INT8**
(MatMul/Gemm — full ConvInteger graphs are not implemented on CoreML).

Start the MinIO registry, then export, score the test slice, and upload:

```bash
docker compose up -d minio          # S3 API :19000  console :19001  (minioadmin / minioadmin)
cd "mot_pipeline (Object_Detection + Tracking)"
python register_variants.py         # writes MinIO bucket mot17-models and registry/catalog.json
python registry.py                  # list name, version, size, HOTA, recall
```

Offline accuracy is TrackEval on **MOT17-02-DPM frames 1–100** (public GT; MOT17
test has none). Detection variants are all tracked with default ByteTrack.
Tracking variants all reuse YOLO26n `imgsz=640` detections.

Every object in MinIO has `metadata.json`: name, version, source, size on disk,
and offline accuracy.

| Stage | Variant | Kind | Size | HOTA | Recall |
|---|---|---|---:|---:|---:|
| detection | `yolo26n-pt-imgsz640` | PyTorch default | 5.3 MB | 36.2 | 25.7 |
| detection | `yolo26n-pt-imgsz1280` | input resolution | 5.3 MB | **43.2** | **39.7** |
| detection | `yolo26n-pt-imgsz640-skip2` | frame skip | 5.3 MB | 15.1 | 7.5 |
| detection | `yolo26n-onnx-imgsz640` | ONNX Runtime | 9.5 MB | 36.8 | 28.3 |
| detection | `yolo26n-onnx-int8-imgsz640` | ORT dynamic INT8 | 9.5 MB | 36.9 | 27.8 |
| tracking | `bytetrack` | default thresholds | 178 B | 36.2 | 25.7 |
| tracking | `bytetrack-low` | lower association thresh | 224 B | 35.2 | 23.4 |
| tracking | `bytetrack-long` | `track_buffer=90` | 208 B | 36.2 | 25.7 |
| tracking | `ocsort` | OC-SORT family | 213 B | 33.5 | 21.7 |

On this test slice, **imgsz=1280** is the only detection change that clearly
improves accuracy. Frame skipping trades recall for FPS. ONNX / INT8 stay within
about 1 HOTA of PyTorch 640. Tracker YAMLs barely move HOTA here (short clip,
few occlusions).

---

## Usage

Run from `mot_pipeline (Object_Detection + Tracking)/`.

### Full 7-sequence sweep (recommended)

```bash
python run_benchmark.py
```

This detects with YOLO26n at 640 / 960 / 1280 and Faster R-CNN MobileNet 320, then
tracks with ByteTrack, ByteTrack-low, and OC-SORT, then evaluates. Resume detections
with `--skip-detect`. Drop Faster R-CNN with `--skip-fasterrcnn`. Restrict YOLO sizes
with `--imgsz 640 1280`.

### Single sequence

```bash
python detect.py --detector yolo26n --sequence MOT17-02-DPM --imgsz 1280
python track.py --tracker bytetrack --sequence MOT17-02-DPM \
  --detections outputs/detections/MOT17-02-DPM/yolo26n_imgsz1280.csv \
  --tracker-config configs/bytetrack_low.yaml
python evaluate.py --sequence MOT17-02-DPM \
  --tracks outputs/tracks/MOT17-02-DPM/yolo26n_imgsz1280_bytetrack_low.txt
```

`--detector` is `yolo26n`, `fasterrcnn` (MobileNet 320) or `detr`. `--tracker` is
`bytetrack` or `ocsort`. `--device` is `auto` (MPS/CUDA/CPU), `cpu`, `mps` or `cuda`.

---

## Motion assertion vs tracking accuracy

For every track, the movement of the box centre between consecutive observations is
measured in image diagonals per frame. A step fails if it exceeds **0.0157**, the 99th
percentile of the same quantity over ground-truth pedestrian tracks in the seven MOT17
training videos.

`evaluate.py` prints the pass rate **next to HOTA, MOTA, IDF1 and Recall**. A 100%
motion pass rate with ~20% recall means the tracks that exist are kinematically
plausible, not that tracking is accurate.

Measured on all seven unique train videos (5,316 frames), **every** configuration
passes **96.9–97.5%** of motion checks — including Faster R-CNN at HOTA 23. The
assertion does not rank detectors. MOT17-05 (handheld, 640×480) is the only sequence
where pass rate drops to the mid-80s; the 1920×1080 videos stay at 99–100%.

| Detector | Tracker | HOTA | MOTA | IDF1 | Recall | Precision | Motion pass | Failed / checked |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n imgsz=1280 | OC-SORT | 41.4 | 44.6 | 51.0 | 53.6 | 87.1 | 96.9 | 1,094 / 69,262 |
| YOLO26n imgsz=1280 | ByteTrack | 41.3 | 44.7 | 50.2 | **55.0** | 85.6 | 97.0 | 1,062 / 73,281 |
| YOLO26n imgsz=960 | ByteTrack | 39.9 | 43.2 | 49.0 | 50.7 | 88.3 | 97.4 | 879 / 60,304 |
| YOLO26n imgsz=640 | ByteTrack | 36.3 | 37.1 | 43.4 | 43.1 | 89.9 | 97.2 | 858 / 43,636 |
| Faster R-CNN MV3-320 | ByteTrack | 23.2 | 8.9 | 26.4 | 35.2 | 56.2 | 97.5 | 870 / 55,253 |

**Chosen configuration: YOLO26n `imgsz=1280` + ByteTrack.** Raising input size is
what recovers small pedestrians. ByteTrack vs OC-SORT is a tie on HOTA; ByteTrack
has the higher recall. Lowering tracker thresholds (`bytetrack_low`) does not help.

Detection recall (IoU 0.5, before tracking) by pedestrian height, all seven videos:

| Height | YOLO 640 | YOLO 1280 | Faster R-CNN MV3-320 |
|---|---:|---:|---:|
| < 40 px | 12.9% | **47.4%** | 2.1% |
| 40–80 px | 39.9% | **61.3%** | 7.4% |
| 80–160 px | 49.0% | 59.4% | 20.4% |
| ≥ 160 px | 74.4% | 80.9% | 52.4% |

The 30-frame MOT17-02 snapshot (23% TrackEval recall at 93% precision) is the small-person
failure mode, not the 7-sequence mean. Full tables:
`mot_pipeline (Object_Detection + Tracking)/results/`.

---

## File formats

Detections CSV. Pixel coordinates in the original image; `class_id` 0 is person:

```text
frame,x1,y1,x2,y2,confidence,class_id
```

Track file. MOTChallenge format, ten columns, no header:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
```

Detectors keep every person box with a score of at least 0.01 (`--conf`). This is only a
floor for collecting candidates: the trackers apply their own confidence thresholds from
the YAML files in `configs/`.
