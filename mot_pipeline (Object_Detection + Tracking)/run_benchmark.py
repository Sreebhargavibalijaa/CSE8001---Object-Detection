"""Run detection, tracking and evaluation on all seven unique MOT17 train videos.

MOT17-*-DPM / FRCNN / SDP folders contain identical frames; this script uses one
copy of each sequence (the DPM folder). YOLO input sizes and tracker thresholds
are profiled because default YOLO26n at imgsz=640 misses many small pedestrians.

Writes per-run files under --output-root and a summary CSV/Markdown under --results-dir.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    UNIQUE_TRAIN_SEQUENCES,
    add_common_arguments,
    load_sequence,
    read_detections,
    resolve_device,
)
from detect import build_detector, run_detection
from evaluate import score_tracks
from track import run_tracking

PIPELINE_DIR = Path(__file__).resolve().parent
HEIGHT_LABELS = ("<40px", "40-80px", "80-160px", ">=160px")
YOLO_IMGSZ = (640, 960, 1280)
TRACKER_CONFIGS = (
    ("bytetrack", PIPELINE_DIR / "configs" / "bytetrack.yaml"),
    ("bytetrack", PIPELINE_DIR / "configs" / "bytetrack_low.yaml"),
    ("ocsort", PIPELINE_DIR / "configs" / "ocsort.yaml"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="7-sequence MOT17 profiling sweep")
    add_common_arguments(parser)
    parser.add_argument("--results-dir", type=Path, default=PIPELINE_DIR / "results")
    parser.add_argument("--imgsz", type=int, nargs="+", default=list(YOLO_IMGSZ))
    parser.add_argument("--skip-detect", action="store_true")
    parser.add_argument("--skip-fasterrcnn", action="store_true",
                        help="Deprecated: use --detectors yolo26n")
    parser.add_argument("--detectors", nargs="+", default=["yolo26n", "fasterrcnn"],
                        choices=["yolo26n", "fasterrcnn"],
                        help="Which detector families to run")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--yolo-weights", type=Path)
    parser.set_defaults(sequence="all")
    return parser.parse_args()


def sequences(args):
    names = UNIQUE_TRAIN_SEQUENCES if args.sequence == "all" else [args.sequence]
    return [load_sequence(args.dataset_root, name) for name in names]


def detection_path(output_root, sequence_name, tag):
    return output_root / "detections" / sequence_name / f"{tag}.csv"


def track_path(output_root, sequence_name, det_tag, tracker_tag):
    return output_root / "tracks" / sequence_name / f"{det_tag}_{tracker_tag}.txt"


def metrics_path(output_root, sequence_name, det_tag, tracker_tag):
    return output_root / "metrics" / sequence_name / f"{det_tag}_{tracker_tag}.csv"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pedestrian_gt(sequence, max_frames=None):
    """MOT17 pedestrians used by TrackEval: class 1, consider flag 1."""
    limit = sequence.frame_limit(max_frames)
    grouped = defaultdict(list)
    gt_path = sequence.directory / "gt" / "gt.txt"
    with gt_path.open(encoding="utf-8") as file:
        for line in file:
            values = line.strip().split(",")
            frame, _, left, top, width, height, consider, class_id = (
                int(float(values[0])), int(float(values[1])),
                float(values[2]), float(values[3]), float(values[4]), float(values[5]),
                int(float(values[6])), int(float(values[7])),
            )
            if frame > limit or consider != 1 or class_id != 1:
                continue
            grouped[frame].append((left, top, left + width, top + height, height))
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


def detection_recall_by_size(sequence, detections, max_frames=None):
    gt = load_pedestrian_gt(sequence, max_frames)
    bins = {label: {"gt": 0, "tp": 0} for label in HEIGHT_LABELS}
    total_gt = total_tp = 0
    limit = sequence.frame_limit(max_frames)
    total_pred = sum(len(detections.get(frame, [])) for frame in range(1, limit + 1))
    for frame, gt_boxes in gt.items():
        pred = [row[:4] for row in detections.get(frame, [])]
        iou = box_iou([box[:4] for box in gt_boxes], pred)
        matched_gt = set()
        if iou.size:
            pairs = sorted((((i, j), iou[i, j]) for i in range(iou.shape[0])
                            for j in range(iou.shape[1]) if iou[i, j] >= 0.5),
                           key=lambda item: item[1], reverse=True)
            used_pred = set()
            for (i, j), _ in pairs:
                if i in matched_gt or j in used_pred:
                    continue
                matched_gt.add(i)
                used_pred.add(j)
        for index, box in enumerate(gt_boxes):
            label = HEIGHT_LABELS[int(np.digitize(box[4], (40, 80, 160)))]
            bins[label]["gt"] += 1
            total_gt += 1
            if index in matched_gt:
                bins[label]["tp"] += 1
                total_tp += 1
    rows = []
    for label in HEIGHT_LABELS:
        gt_count = bins[label]["gt"]
        tp = bins[label]["tp"]
        rows.append({
            "sequence": sequence.name,
            "height_bin": label,
            "gt": gt_count,
            "tp": tp,
            "recall": 100.0 * tp / gt_count if gt_count else None,
        })
    precision = 100.0 * total_tp / total_pred if total_pred else None
    return rows, {
        "sequence": sequence.name,
        "gt": total_gt,
        "tp": total_tp,
        "detections": total_pred,
        "recall": 100.0 * total_tp / total_gt if total_gt else None,
        "precision": precision,
    }


def release_detector(detect):
    """Drop detector weights before the next model so MPS/CPU RAM is not stacked."""
    del detect
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_detectors(args, seqs, tags):
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    for detector, imgsz, tag in tags:
        print(f"\n=== detect {tag} ===", flush=True)
        _, detect = build_detector(detector, args.conf, device, imgsz, args.yolo_weights)
        try:
            for sequence in seqs:
                output = detection_path(args.output_root, sequence.name, tag)
                if args.skip_detect and output.is_file():
                    print(f"skip existing {output}")
                    continue
                run_detection(sequence, detect, output, args.max_frames, tag)
        finally:
            release_detector(detect)


def run_trackers_and_eval(args, seqs, det_tags):
    results = []
    size_rows = []
    size_summary = []
    for sequence in seqs:
        for det_tag in det_tags:
            det_file = detection_path(args.output_root, sequence.name, det_tag)
            detections = read_detections(det_file)
            per_seq_rows, overall = detection_recall_by_size(sequence, detections, args.max_frames)
            for row in per_seq_rows:
                row["detector"] = det_tag
            overall["detector"] = det_tag
            size_rows.extend(per_seq_rows)
            size_summary.append(overall)
            print(f"{sequence.name} {det_tag} det-recall={overall['recall']:.1f}% "
                  f"det-precision={overall['precision']:.1f}%", flush=True)
            for tracker_name, config in TRACKER_CONFIGS:
                tracker_tag = config.stem
                tracks_file = track_path(args.output_root, sequence.name, det_tag, tracker_tag)
                print(f"track {sequence.name} {det_tag} + {tracker_tag}", flush=True)
                run_tracking(sequence, detections, tracker_name, config, tracks_file, args.max_frames)
                result = score_tracks(sequence, tracks_file, args.max_frames)
                result["detector"] = det_tag
                result["tracker"] = tracker_tag
                metrics_file = metrics_path(args.output_root, sequence.name, det_tag, tracker_tag)
                write_csv(metrics_file, [result])
                pass_rate = result["assertion_pass_rate"]
                print(f"  HOTA {result['HOTA']:.1f}  MOTA {result['MOTA']:.1f}  "
                      f"IDF1 {result['IDF1']:.1f}  Rec {result['Recall']:.1f}  "
                      f"motion {pass_rate:.1f}% ({int(result['assertions_failed'])}/"
                      f"{int(result['assertions_checked'])} failed)", flush=True)
                results.append(result)
    return results, size_rows, size_summary


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["detector"], row["tracker"])].append(row)
    summary = []
    for (detector, tracker), rows in grouped.items():
        item = {"detector": detector, "tracker": tracker, "sequences": len(rows)}
        for key in ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "Precision", "Recall",
                    "assertion_pass_rate"):
            item[key] = mean([row[key] for row in rows])
        for key in ("FP", "FN", "IDSW", "Frag", "assertions_checked", "assertions_failed"):
            item[key] = int(sum(row[key] for row in rows))
        item["frames"] = int(sum(row["frames"] for row in rows))
        summary.append(item)
    summary.sort(key=lambda row: (-(row["HOTA"] or 0), -(row["Recall"] or 0)))
    return summary


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.1f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def write_report(results_dir, results, summary, size_rows, size_summary):
    write_csv(results_dir / "per_sequence.csv", results)
    write_csv(results_dir / "summary.csv", summary)
    write_csv(results_dir / "detection_recall_by_size.csv", size_rows)
    write_csv(results_dir / "detection_recall.csv", size_summary)
    tracking_cols = ["detector", "tracker", "HOTA", "MOTA", "IDF1", "Recall", "Precision",
                     "assertion_pass_rate", "assertions_failed", "assertions_checked"]
    motion_cols = ["sequence", "detector", "tracker", "HOTA", "MOTA", "IDF1", "Recall",
                   "assertion_pass_rate", "assertions_failed"]
    report = [
        "# MOT17 seven-sequence profiling",
        "",
        "One copy of each training video (`*-DPM`). DPM / FRCNN / SDP folders share frames.",
        "",
        "## Configuration means (unweighted over 7 sequences)",
        "",
        markdown_table(summary, tracking_cols),
        "",
        "Motion-assertion pass rate is reported next to HOTA / MOTA / IDF1 / Recall so a",
        "kinematically smooth tracker is not mistaken for an accurate one. High pass rate",
        "with low Recall means the tracks that exist move like people, but most pedestrians",
        "were never detected.",
        "",
        "## Per-sequence motion assertion vs tracking accuracy",
        "",
        markdown_table(results, motion_cols),
        "",
        "## Detection recall by pedestrian height (IoU 0.5, before tracking)",
        "",
        markdown_table([row for row in size_rows if row["gt"]],
                       ["detector", "sequence", "height_bin", "gt", "tp", "recall"]),
        "",
    ]
    path = results_dir / "REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSummary -> {results_dir / 'summary.csv'}")
    print(f"Report  -> {path}")
    print("\n" + markdown_table(summary, tracking_cols))


def main():
    args = parse_args()
    seqs = sequences(args)
    wanted = set(args.detectors)
    if args.skip_fasterrcnn:
        wanted.discard("fasterrcnn")
    tags = []
    if "yolo26n" in wanted:
        tags.extend(("yolo26n", imgsz, f"yolo26n_imgsz{imgsz}") for imgsz in args.imgsz)
    if "fasterrcnn" in wanted:
        tags.append(("fasterrcnn", None, "fasterrcnn_mv3_320"))
    if not tags:
        raise SystemExit("No detectors selected")
    print("Sequences:", ", ".join(sequence.name for sequence in seqs))
    print("Detectors:", ", ".join(tag for _, _, tag in tags))
    run_detectors(args, seqs, tags)
    results, size_rows, size_summary = run_trackers_and_eval(
        args, seqs, [tag for _, _, tag in tags])
    summary = aggregate(results)
    write_report(args.results_dir, results, summary, size_rows, size_summary)


if __name__ == "__main__":
    main()
