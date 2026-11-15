"""Score MOT tracks with Guide v4 metrics and online assertions.

Primary offline accuracy: TrackEval HOTA (MOT17 pedestrian settings).
Also: MOTA, IDF1, ID switches, detector precision/recall at IoU 0.5.

Required assertions are per-frame and need no labels. A frame passes when A1–A3
all hold. Milestone 1 used a different motion check (centre jump / image diagonal);
that number is still reported as m1_motion_pass_rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trackeval

from common import add_common_arguments, load_sequence, read_detections, read_tracks

# Milestone 1 motion threshold: 99th percentile of GT centre movement / image
# diagonal / frame, fit on all seven MOT17 train videos (includes data the guide
# now holds out as the test slice — do not use this to tune).
M1_MOTION_THRESHOLD = 0.01570218735615795
MOTION_THRESHOLD = M1_MOTION_THRESHOLD  # kept for callers that import the old name


def tracking_metrics(sequence, tracks_path, max_frames=None):
    """Percentages except FP, FN, IDSW and Frag. MOTP is mean matched IoU."""
    flat, _ = tracking_metrics_raw(sequence, tracks_path, max_frames)
    return flat


def tracking_metrics_raw(sequence, tracks_path, max_frames=None):
    limit = sequence.frame_limit(max_frames)
    with tempfile.TemporaryDirectory(prefix="trackeval-") as stage:
        prediction = Path(stage) / "input" / "data" / f"{sequence.name}.txt"
        prediction.parent.mkdir(parents=True)
        shutil.copyfile(tracks_path, prediction)
        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": str(sequence.directory.parent),
            "TRACKERS_FOLDER": stage,
            "OUTPUT_FOLDER": stage,
            "TRACKERS_TO_EVAL": ["input"],
            "TRACKER_SUB_FOLDER": "data",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {sequence.name: sequence.seq_length},
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        })
        raw = dataset.get_raw_seq_data("input", sequence.name)
        for key, value in raw.items():
            if isinstance(value, list):
                raw[key] = value[:limit]
        raw["num_timesteps"] = limit
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
    hota, clear, identity = (
        getattr(trackeval.metrics, name)({"PRINT_CONFIG": False}).eval_sequence(data)
        for name in ("HOTA", "CLEAR", "Identity")
    )
    return flatten_trackeval(hota, clear, identity), {"HOTA": hota, "CLEAR": clear, "Identity": identity}


def flatten_trackeval(hota, clear, identity):
    metrics = {key: float(np.mean(hota[key]) * 100) for key in ("HOTA", "DetA", "AssA", "LocA")}
    metrics.update({key: float(clear[key] * 100) for key in ("MOTA", "MOTP")})
    metrics.update({key: float(identity[key] * 100) for key in ("IDF1", "IDP", "IDR")})
    metrics.update({
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "Precision": float(clear["CLR_Pr"] * 100),
        "Recall": float(clear["CLR_Re"] * 100),
    })
    return metrics


def combine_trackeval(per_sequence_raw):
    """Official TrackEval combination across sequences (not an unweighted mean)."""
    names = ("HOTA", "CLEAR", "Identity")
    combined = {}
    for name in names:
        metric = getattr(trackeval.metrics, name)({"PRINT_CONFIG": False})
        combined[name] = metric.combine_sequences(
            {seq: raw[name] for seq, raw in per_sequence_raw.items()}
        )
    return flatten_trackeval(combined["HOTA"], combined["CLEAR"], combined["Identity"])


def pedestrian_gt(sequence, max_frames=None):
    """Ground-truth rows with class 1 (pedestrian) and consider-flag 1."""
    limit = sequence.frame_limit(max_frames)
    grouped = defaultdict(list)
    with (sequence.directory / "gt" / "gt.txt").open(encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split(",")
            frame = int(float(values[0]))
            if frame > limit:
                continue
            left, top, width, height = map(float, values[2:6])
            consider, class_id = int(float(values[6])), int(float(values[7]))
            if consider == 1 and class_id == 1:
                grouped[frame].append((left, top, left + width, top + height))
    return grouped


def box_iou(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    g = np.asarray(gt, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    inter_x1 = np.maximum(g[:, 0:1], p[:, 0])
    inter_y1 = np.maximum(g[:, 1:2], p[:, 1])
    inter_x2 = np.minimum(g[:, 2:3], p[:, 2])
    inter_y2 = np.minimum(g[:, 3:4], p[:, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return inter / np.clip(area_g[:, None] + area_p - inter, 1e-6, None)


def greedy_matches(iou, threshold=0.5):
    matched_gt, matched_pred = set(), set()
    if not iou.size:
        return matched_gt, matched_pred
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
         if iou[i, j] >= threshold),
        reverse=True,
    )
    for _, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
    return matched_gt, matched_pred


def detection_precision_recall(sequence, detections, max_frames=None):
    """Person detection P/R at IoU ≥ 0.5 vs GT class 1, consider-flag 1."""
    gt = pedestrian_gt(sequence, max_frames)
    limit = sequence.frame_limit(max_frames)
    tp = fp = fn = 0
    for frame in range(1, limit + 1):
        gt_boxes = gt.get(frame, [])
        pred = [row[:4] for row in detections.get(frame, [])]
        iou = box_iou(gt_boxes, pred)
        matched_gt, matched_pred = greedy_matches(iou, 0.5)
        tp += len(matched_gt)
        fn += len(gt_boxes) - len(matched_gt)
        fp += len(pred) - len(matched_pred)
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else None
    recall = 100.0 * tp / (tp + fn) if (tp + fn) else None
    return {
        "det_tp": tp, "det_fp": fp, "det_fn": fn,
        "det_precision": precision, "det_recall": recall,
    }


def required_assertions(sequence, tracks, max_frames=None):
    """Guide v4 A1–A3, plus extra A4. Pass rate is the fraction of frames that pass all required checks."""
    limit = sequence.frame_limit(max_frames)
    by_frame = defaultdict(dict)
    for frame, track_id, left, top, width, height, _ in tracks:
        if frame > limit:
            continue
        by_frame[frame][track_id] = (left + width / 2, top + height / 2, width, height, left, top)

    prev = {}
    prev_count = None
    counts = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "required": 0}
    for frame in range(1, limit + 1):
        boxes = by_frame.get(frame, {})
        a1 = a2 = a3 = a4 = True
        for track_id, (cx, cy, width, height, left, top) in boxes.items():
            if width <= 0 or not (1.0 <= height / width <= 6.0):
                a2 = False
            right, bottom = left + width, top + height
            if right < 0 or bottom < 0 or left > sequence.width or top > sequence.height:
                a4 = False
            if track_id in prev:
                pcx, pcy = prev[track_id][:2]
                if math.hypot(cx - pcx, cy - pcy) > 0.5 * height:
                    a1 = False
        count = len(boxes)
        if prev_count is not None and abs(count - prev_count) > max(3, 0.3 * prev_count):
            a3 = False
        counts["A1"] += a1
        counts["A2"] += a2
        counts["A3"] += a3
        counts["A4"] += a4
        counts["required"] += a1 and a2 and a3
        prev = boxes
        prev_count = count

    def rate(passed):
        return 100.0 * passed / limit if limit else None

    return {
        "assertions_checked": limit,
        "assertions_failed": limit - counts["required"],
        "assertion_pass_rate": rate(counts["required"]),
        "A1_pass_rate": rate(counts["A1"]),
        "A2_pass_rate": rate(counts["A2"]),
        "A3_pass_rate": rate(counts["A3"]),
        "A4_pass_rate": rate(counts["A4"]),
    }


def m1_motion_assertion(sequence, tracks, threshold=M1_MOTION_THRESHOLD, max_frames=None):
    """Milestone 1 motion check: centre jump / image diagonal / frame gap."""
    limit = sequence.frame_limit(max_frames)
    centres = defaultdict(list)
    for frame, track_id, left, top, width, height, _ in tracks:
        if frame <= limit:
            centres[track_id].append((frame, left + width / 2, top + height / 2))
    diagonal = math.hypot(sequence.width, sequence.height)
    checked = passed = 0
    for observations in centres.values():
        ordered = sorted(observations)
        for first, second in zip(ordered, ordered[1:]):
            speed = math.hypot(second[1] - first[1], second[2] - first[2]) / diagonal / (second[0] - first[0])
            checked += 1
            passed += speed <= threshold
    return {
        "m1_assertions_checked": checked,
        "m1_assertions_failed": checked - passed,
        "m1_motion_pass_rate": 100.0 * passed / checked if checked else None,
    }


def motion_assertion(sequence, tracks, threshold=M1_MOTION_THRESHOLD, max_frames=None):
    """Required Guide v4 assertions. `threshold` is ignored; kept for old callers."""
    del threshold
    required = required_assertions(sequence, tracks, max_frames)
    return {
        "assertions_checked": required["assertions_checked"],
        "assertions_failed": required["assertions_failed"],
        "assertion_pass_rate": required["assertion_pass_rate"],
    }


def score_tracks(sequence, tracks_path, max_frames=None, motion_threshold=M1_MOTION_THRESHOLD,
                 detections=None):
    result, _ = score_tracks_with_raw(sequence, tracks_path, max_frames, motion_threshold, detections)
    return result


def score_tracks_with_raw(sequence, tracks_path, max_frames=None, motion_threshold=M1_MOTION_THRESHOLD,
                          detections=None):
    limit = sequence.frame_limit(max_frames)
    tracks = [row for row in read_tracks(tracks_path) if row[0] <= limit]
    flat, raw = tracking_metrics_raw(sequence, tracks_path, max_frames)
    result = {
        "sequence": sequence.name,
        "tracks": Path(tracks_path).stem,
        "frames": limit,
        **flat,
        **required_assertions(sequence, tracks, max_frames),
        **m1_motion_assertion(sequence, tracks, motion_threshold, max_frames),
    }
    if detections is not None:
        result.update(detection_precision_recall(sequence, detections, max_frames))
    return result, raw


def main():
    parser = argparse.ArgumentParser(description="Guide v4 TrackEval + online assertions")
    add_common_arguments(parser)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--detections", type=Path, help="Optional detections CSV for detector P/R")
    parser.add_argument("--motion-threshold", type=float, default=M1_MOTION_THRESHOLD)
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    detections = read_detections(args.detections) if args.detections else None
    result = score_tracks(sequence, args.tracks, args.max_frames, args.motion_threshold, detections)
    for key, value in result.items():
        print(f"{key:>22}: {value:.3f}" if isinstance(value, float) else f"{key:>22}: {value}")
    print(f"{'guide vs tracking':>22}: A1–A3 pass {result['assertion_pass_rate']:.1f}% | "
          f"HOTA {result['HOTA']:.1f} | MOTA {result['MOTA']:.1f} | "
          f"IDF1 {result['IDF1']:.1f} | IDSW {result['IDSW']}")
    output = args.output_root / "metrics" / sequence.name / f"{args.tracks.stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
