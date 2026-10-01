# MOT17 Object Detection & Video Generation Pipeline

A computer vision workflow for evaluating and comparing object detection models on the **MOT17** (Multiple Object Tracking) benchmark dataset, with automated annotated frame generation and video synthesis.

---

## Table of Contents

- [Overview](#overview)
- [Workspace Structure](#workspace-structure)
- [Requirements & Installation](#requirements--installation)
- [Dataset Organization](#dataset-organization)
- [Model Pipelines](#model-pipelines)
  - [1. Ultralytics YOLO (`video_processing.py`)](#1-ultralytics-yolo-video_processingpy)
  - [2. Torchvision Faster R-CNN with ResNet-50 FPN (`facebook_fpn.py`)](#2-torchvision-faster-r-cnn-with-resnet-50-fpn-facebook_fpnpy)
  - [3. Hugging Face DETR ResNet-50 (`facebookresnet.py`)](#3-hugging-face-detr-resnet-50-facebookresnetpy)
  - [4. Video Creation (`create_video.py`)](#4-video-creation-create_videopy)
  - [5. ByteTrack Tracking (`bytetrack_tracking.py`)](#5-bytetrack-tracking-bytetrack_trackingpy)
- [Step-by-Step Usage](#step-by-step-usage)
- [Configuration & Customization](#configuration--customization)

---

## Overview

This repository provides scripts to run object detection inference across MOT17 video sequences, overlay bounding boxes on individual frames, and render the resulting sequence into an MP4 video.

Supported detection architectures:
- **YOLO** (via Ultralytics, using `yolo26n.pt`)
- **Faster R-CNN with ResNet-50 FPN** (via `torchvision.models.detection`)
- **DETR ResNet-50** (End-to-End Object Detection with Transformers via Hugging Face)

---

## Workspace Structure

```text
8001/
├── MOT17/                         # MOT17 benchmark dataset
│   ├── train/                     # Training sequences (MOT17-02-*, MOT17-04-*, etc.)
│   │   └── MOT17-02-DPM/
│   │       ├── img1/              # Original frame images (.jpg)
│   │       ├── det/               # Public detection baselines
│   │       ├── gt/                # Ground truth annotations
│   │       ├── seqinfo.ini        # Sequence metadata (framerate, resolution, etc.)
│   │       ├── yolo_boxes/        # Output annotated frames from YOLO
│   │       ├── fb_resnet_boxes/   # Output annotated frames from ResNet models
│   │       ├── yolo_detections.csv    # Raw YOLO person detections (tracker input)
│   │       ├── bytetrack_boxes/       # Output frames with ByteTrack IDs drawn
│   │       └── bytetrack_results.txt  # MOT-format tracking results
│   └── test/                      # Test sequences
├── create_video.py                # Compiles image sequences into an MP4 video
├── facebook_fpn.py                # Faster R-CNN ResNet-50 FPN detection pipeline
├── facebookresnet.py              # Hugging Face DETR ResNet-50 detection pipeline
├── video_processing.py            # YOLO detection pipeline
├── bytetrack_tracking.py          # ByteTrack tracking over saved detections
├── requirements.txt               # Python dependencies
├── yolo26n.pt                     # YOLO model weights
└── README.md                      # Project documentation
```

---

## Requirements & Installation

Ensure Python 3.8+ is installed. Install the necessary dependencies:

```bash
pip install torch torchvision transformers ultralytics opencv-python
```

Or install everything, including the tracking dependencies, from the pinned list:

```bash
pip install -r requirements.txt
```

### Key Libraries
- **PyTorch** & **Torchvision**: Deep learning framework and vision models (`fasterrcnn_resnet50_fpn`).
- **Transformers**: Hugging Face library for transformer-based vision models (`facebook/detr-resnet-50`).
- **Ultralytics**: YOLO detection engine and utilities, plus the ByteTrack implementation.
- **OpenCV (`cv2`)**: Image reading, drawing bounding boxes, and video encoding (`VideoWriter`).
- **lap**: Linear assignment solver used by ByteTrack to associate detections with tracks.

---

## Dataset Organization

Download and extract the MOT17 dataset into the `MOT17/` folder:

```text
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── seqinfo.ini
│   └── ...
└── test/
```

Each sequence folder contains an `img1/` directory with ordered image frames (`000001.jpg`, `000002.jpg`, ...).

---

## Model Pipelines

### 1. Ultralytics YOLO (`video_processing.py`)
- **Model**: `yolo26n.pt`
- **Output Directory**: `<sequence_path>/yolo_boxes/`
- **Description**: Loads the local YOLO model weights, iterates through each sequence in `MOT17/train/`, detects objects, plots bounding boxes, and saves annotated frames.

Run YOLO inference:
```bash
python3 video_processing.py
```

### 2. Torchvision Faster R-CNN with ResNet-50 FPN (`facebook_fpn.py`)
- **Model**: Pretrained `fasterrcnn_resnet50_fpn` with `FasterRCNN_ResNet50_FPN_Weights.DEFAULT`
- **Confidence Threshold**: 0.50
- **Output Directory**: `<sequence_path>/fb_resnet_boxes/`
- **Description**: Converts images to normalized tensors, executes inference in evaluation mode (`torch.no_grad()`), and draws green bounding boxes with class labels and confidence scores.

Run Faster R-CNN inference:
```bash
python3 facebook_fpn.py
```

### 3. Hugging Face DETR ResNet-50 (`facebookresnet.py`)
- **Model**: `facebook/detr-resnet-50` (via Hugging Face `transformers`)
- **Output Directory**: `<sequence_path>/fb_resnet_boxes/`
- **Description**: Applies transformer-based object detection, plots detections, and saves annotated frames to disk.

Run DETR ResNet-50 inference:
```bash
python3 facebookresnet.py
```

### 4. Video Creation (`create_video.py`)
Compiles a sorted sequence of images from a directory into an MP4 video file using OpenCV's `mp4v` codec.

CLI Arguments:
- `--image_dir`: Path to the folder containing image frames.
- `--output`: Destination path for the output `.mp4` video.
- `--fps`: Target frame rate (default: `30`).

### 5. ByteTrack Tracking (`bytetrack_tracking.py`)
- **Tracker**: ByteTrack (`bytetrack.yaml`), via the implementation bundled with Ultralytics
- **Input**: `<sequence_path>/yolo_detections.csv`, written by `video_processing.py`
- **Output Directory**: `<sequence_path>/bytetrack_boxes/`
- **Output File**: `<sequence_path>/bytetrack_results.txt`
- **Description**: Reads saved detections and associates them across frames, assigning each person a stable track ID. The detector is **not** re-run here, so tracking can be re-tuned in seconds without repeating inference. Track IDs are drawn on the original MOT17 frames, each in its own color.

`video_processing.py` writes one row per detected person:

```text
frame,x1,y1,x2,y2,confidence,class_id
```

`bytetrack_tracking.py` writes standard MOT-format results, ready for evaluation against the `gt/` annotations:

```text
frame,id,x,y,width,height,confidence,-1,-1,-1
```

Run ByteTrack:
```bash
python3 bytetrack_tracking.py
```

CLI Arguments:
- `--sequence_dir`: MOT17 sequence folder containing `img1/` (default: `MOT17/train/MOT17-02-DPM`).
- `--detections`: Detections CSV (default: `<sequence_dir>/yolo_detections.csv`).
- `--tracker`: Tracker config (default: `bytetrack.yaml`).
- `--output_dir`: Where to write tracked frames (default: `<sequence_dir>/<tracker>_boxes`).
- `--output_txt`: MOT-format results file (default: `<sequence_dir>/<tracker>_results.txt`).

Because outputs are named after the tracker, swapping trackers needs no code changes and will not overwrite a previous run:

```bash
# OC-SORT instead of ByteTrack -> ocsort_boxes/ and ocsort_results.txt
python3 bytetrack_tracking.py --tracker ocsort.yaml
```

Any detector can feed the tracker by writing the same CSV columns and pointing `--detections` at it.

---

## Step-by-Step Usage

### Step 1: Run Detection on Sequence Frames
Choose a detector to generate annotated image frames:

```bash
# Option A: Run YOLO detection
python3 video_processing.py

# Option B: Run Faster R-CNN ResNet-50 FPN detection
python3 facebook_fpn.py

# Option C: Run DETR ResNet-50 detection
python3 facebookresnet.py
```

### Step 2: Track Detections Across Frames (optional)
Assign stable IDs to the people YOLO detected in Step 1:

```bash
python3 bytetrack_tracking.py
```

This reads `yolo_detections.csv` and produces `bytetrack_boxes/` plus `bytetrack_results.txt`.

### Step 3: Compile Annotated Frames into Video
Convert the generated image sequence into an MP4 video:

```bash
# Example for YOLO-annotated frames
python3 create_video.py \
  --image_dir "MOT17/train/MOT17-02-DPM/yolo_boxes" \
  --output "MOT17/train/MOT17-02-DPM/yolo_boxes_video.mp4" \
  --fps 30

# Example for ResNet/DETR-annotated frames
python3 create_video.py \
  --image_dir "MOT17/train/MOT17-02-DPM/fb_resnet_boxes" \
  --output "MOT17/train/MOT17-02-DPM/fb_resnet_boxes_video.mp4" \
  --fps 30

# Example for ByteTrack-annotated frames
python3 create_video.py \
  --image_dir "MOT17/train/MOT17-02-DPM/bytetrack_boxes" \
  --output "MOT17/train/MOT17-02-DPM/bytetrack_video.mp4" \
  --fps 30
```

---

## Configuration & Customization

- **Dataset Path**: Update `base_path` in `video_processing.py`, `facebook_fpn.py`, or `facebookresnet.py` if your dataset is stored in a different location.
- **Confidence Threshold**: Adjust the confidence cutoff in `facebook_fpn.py` (default: `0.50`) to filter out false positives or retain faint detections.
- **Frame Rate**: Match the `--fps` argument in `create_video.py` to the frame rate specified in `seqinfo.ini` for each sequence (e.g., 30 FPS for MOT17-02).
- **Sequences**: `video_processing.py` processes the sequences listed in its `sequences` variable (default: `MOT17-02-DPM`). Add sequence names to that list to run on more of the dataset.
- **Tracker Behaviour**: ByteTrack thresholds live in Ultralytics' `bytetrack.yaml` (`track_high_thresh`, `track_buffer`, `match_thresh`). Copy it locally and pass it with `--tracker path/to/your.yaml` to tune association without editing the package.
