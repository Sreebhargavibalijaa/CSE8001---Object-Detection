# MOT17 Detection + Tracking + Evaluation Pipeline

Compares three pretrained person detectors with two trackers on MOT17 and scores each
combination with the official TrackEval metrics.

```text
MOT17 images -> detect.py -> detections CSV -> track.py -> MOT track file -> evaluate.py -> metrics
```

| Stage | Options |
|---|---|
| Detector | YOLO26n, Faster R-CNN ResNet50-FPN (torchvision), DETR ResNet-50 (`facebook/detr-resnet-50`) |
| Tracker | ByteTrack, OC-SORT (the implementations bundled with Ultralytics) |
| Metrics | HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, IDP, IDR, FP, FN, IDSW, Frag, Precision, Recall |

Everything runs on CPU with pretrained weights. No training or fine-tuning is involved.

---

## Files

| File | What it does | Output |
|---|---|---|
| `detect.py` | Runs one detector over a sequence and keeps the person boxes | `outputs/detections/<sequence>/<detector>.csv` |
| `track.py` | Runs ByteTrack or OC-SORT over those saved boxes | `outputs/tracks/<sequence>/<detector>_<tracker>.txt` |
| `evaluate.py` | TrackEval metrics against the ground truth, plus a motion assertion | `outputs/metrics/<sequence>/<detector>_<tracker>.csv` |
| `common.py` | Helpers shared by the three scripts | |

Each script reads the file the previous one saved. A detector therefore runs once, and both
trackers work from exactly the same detections.

---

## Setup

**1. Install the dependencies** from the repository root (`trackeval` is needed by `evaluate.py`):

```bash
pip install -r requirements.txt trackeval
```

**2. Place the MOT17 dataset** in the repository root, as described in the main README:

```text
CSE8001---Object-Detection/
├── MOT17/
│   └── train/
│       ├── MOT17-02-DPM/   (img1/, gt/, seqinfo.ini)
│       └── ...
└── mot_pipeline/
```

If the dataset is somewhere else, add `--dataset-root <path to MOT17>` to every command.

Model weights are downloaded automatically the first time each detector is used
(about 5 MB for YOLO26n, about 160 MB each for Faster R-CNN and DETR).

---

## Usage

Run the commands from this `mot_pipeline/` folder. The examples use the first 30 frames of
MOT17-02-DPM as a quick test. Remove `--max-frames 30` to process the whole video.

### Step 1: Detect

```bash
python detect.py --detector yolo26n --sequence MOT17-02-DPM --max-frames 30
```

`--detector` can be `yolo26n`, `fasterrcnn` or `detr`. The CSV is named after the detector.

### Step 2: Track

```bash
python track.py --tracker bytetrack --sequence MOT17-02-DPM --max-frames 30 --detections outputs/detections/MOT17-02-DPM/yolo26n.csv --video outputs/videos/yolo26n_bytetrack.mp4
```

`--tracker` can be `bytetrack` or `ocsort`. `--video` is optional and renders the original frames
with a coloured box and ID for every track.

### Step 3: Evaluate

```bash
python evaluate.py --sequence MOT17-02-DPM --max-frames 30 --tracks outputs/tracks/MOT17-02-DPM/yolo26n_bytetrack.txt
```

Prints the metrics and saves them as a one-row CSV.

### Options shared by all three scripts

| Option | Default | Meaning |
|---|---|---|
| `--sequence` | `MOT17-02-DPM` | Any folder name under `MOT17/train` |
| `--max-frames` | whole video | Only use frames 1 through N |
| `--dataset-root` | `../MOT17` | Location of the MOT17 folder |
| `--output-root` | `outputs` | Where results are written |

Existing output files are overwritten.

### Speed

On CPU, YOLO26n takes about 0.1 s per frame, DETR 1.5-4 s and Faster R-CNN 3-8 s.
A full 600-frame video therefore takes about a minute with YOLO26n and up to an hour or more
with the other two.

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

Detectors keep every person box with a score of at least 0.01 (`--conf`). This is only a floor
for collecting candidates: the trackers apply their own confidence thresholds from the
Ultralytics tracker YAML files. A custom YAML can be passed to `track.py` with `--tracker-config`.

---

## Evaluation details

- **TrackEval** is run with the official MOT17 settings: `MotChallenge2DBox`, preprocessing on,
  pedestrian class, and the HOTA, CLEAR and Identity metric families.
- All values are percentages except FP, FN, IDSW and Frag, which are counts. MOTP is the mean
  IoU of matched boxes.
- **Motion assertion**: for every track, the movement of the box centre between consecutive
  observations is measured in image diagonals per frame. A step fails if it exceeds 0.0157,
  the 99th percentile of the same quantity over ground-truth pedestrian tracks in the seven
  MOT17 training videos. `evaluate.py` reports how many steps were checked and the pass rate.
  The threshold can be changed with `--motion-threshold`.
