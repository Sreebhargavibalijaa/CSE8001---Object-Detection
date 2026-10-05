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

---

## Results

Full MOT17-02-DPM (600 frames) and MOT17-09-DPM (525 frames), combined.
FPS is detector plus tracker on CPU with 4 threads.

| Detector | Tracker | HOTA | MOTA | IDF1 | Motion assertion pass % | FPS |
|---|---|---:|---:|---:|---:|---:|
| YOLO26n | ByteTrack | 31.12 | **26.95** | 34.39 | 99.14 | **4.17** |
| YOLO26n | OC-SORT | **31.87** | 26.86 | **36.07** | 99.17 | 4.15 |
| Faster R-CNN | ByteTrack | 31.72 | 18.14 | 36.04 | 99.51 | 0.13 |
| Faster R-CNN | OC-SORT | 30.68 | 17.36 | 35.62 | 99.45 | 0.13 |
| DETR | ByteTrack | 28.02 | 4.80 | 28.87 | 99.38 | 0.24 |
| DETR | OC-SORT | 26.69 | 2.15 | 26.66 | 99.32 | 0.24 |

Per video:

| Detector | Tracker | 02 HOTA | 09 HOTA | 02 MOTA | 09 MOTA | 02 IDF1 | 09 IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | ByteTrack | 26.18 | 43.39 | 18.98 | 54.76 | 27.08 | 52.71 |
| YOLO26n | OC-SORT | 27.35 | 43.30 | 18.87 | 54.72 | 29.04 | 53.71 |
| Faster R-CNN | ByteTrack | 27.57 | 43.30 | 12.13 | 39.12 | 31.75 | 49.32 |
| Faster R-CNN | OC-SORT | 27.64 | 39.59 | 11.49 | 37.82 | 32.09 | 46.60 |
| DETR | ByteTrack | 23.47 | 39.27 | -0.73 | 24.11 | 23.98 | 42.94 |
| DETR | OC-SORT | 21.53 | 38.90 | -3.30 | 21.16 | 21.25 | 42.25 |

These are two training videos, not the MOT17 test set, so the ranking is descriptive only.
The results were produced with `ultralytics` 8.4.63 and `trackeval` 1.3.0; other versions may
give slightly different numbers.
