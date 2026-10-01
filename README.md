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
│   │       └── fb_resnet_boxes/   # Output annotated frames from ResNet models
│   └── test/                      # Test sequences
├── create_video.py                # Compiles image sequences into an MP4 video
├── facebook_fpn.py                # Faster R-CNN ResNet-50 FPN detection pipeline
├── facebookresnet.py              # Hugging Face DETR ResNet-50 detection pipeline
├── video_processing.py            # YOLO detection pipeline
├── yolo26n.pt                     # YOLO model weights
└── README.md                      # Project documentation
```

---

## Requirements & Installation

Ensure Python 3.8+ is installed. Install the necessary dependencies:

```bash
pip install torch torchvision transformers ultralytics opencv-python
```

### Key Libraries
- **PyTorch** & **Torchvision**: Deep learning framework and vision models (`fasterrcnn_resnet50_fpn`).
- **Transformers**: Hugging Face library for transformer-based vision models (`facebook/detr-resnet-50`).
- **Ultralytics**: YOLO detection engine and utilities.
- **OpenCV (`cv2`)**: Image reading, drawing bounding boxes, and video encoding (`VideoWriter`).

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

### Step 2: Compile Annotated Frames into Video
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
```

---

## Configuration & Customization

- **Dataset Path**: Update `base_path` in `video_processing.py`, `facebook_fpn.py`, or `facebookresnet.py` if your dataset is stored in a different location.
- **Confidence Threshold**: Adjust the confidence cutoff in `facebook_fpn.py` (default: `0.50`) to filter out false positives or retain faint detections.
- **Frame Rate**: Match the `--fps` argument in `create_video.py` to the frame rate specified in `seqinfo.ini` for each sequence (e.g., 30 FPS for MOT17-02).
