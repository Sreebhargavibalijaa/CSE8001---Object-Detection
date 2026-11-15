"""Evaluate Milestone 1 configs on the Guide v4 test slice only.

Development slice is used only to record IDs (assertion thresholds A1–A3 are
fixed by the guide). Results, including the ByteTrack+public-FRCNN baseline and
the gold-input ablation, are written under results/guide_eval/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from common import DETECTION_COLUMNS, load_sequence, read_detections, write_detections
from evaluate import combine_trackeval, score_tracks_with_raw
from slices import GUIDE_SEQUENCES, load_manifest, slice_dataset_root
from track import run_tracking

PIPELINE = Path(__file__).resolve().parent
TRACKERS = (
    ("bytetrack", PIPELINE / "configs" / "bytetrack.yaml", "bytetrack"),
    ("bytetrack", PIPELINE / "configs" / "bytetrack_low.yaml", "bytetrack_low"),
    ("ocsort", PIPELINE / "configs" / "ocsort.yaml", "ocsort"),
)
M1_DETECTORS = ("yolo26n_imgsz640", "yolo26n_imgsz960", "yolo26n_imgsz1280", "fasterrcnn_mv3_320")
DPM_NAME = {
    "MOT17-02-FRCNN": "MOT17-02-DPM",
    "MOT17-04-FRCNN": "MOT17-04-DPM",
    "MOT17-09-FRCNN": "MOT17-09-DPM",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Guide v4 test-slice evaluation harness")
    parser.add_argument("--slice", default="test", choices=["test", "development"])
    parser.add_argument("--results-dir", type=Path, default=PIPELINE / "results" / "guide_eval")
    parser.add_argument("--full-detections", type=Path, default=PIPELINE / "outputs" / "detections")
    parser.add_argument("--output-root", type=Path, default=PIPELINE / "outputs" / "guide")
    return parser.parse_args()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def remap_full_detections(src, dest, start, end):
    offset = start - 1
    grouped = read_detections(src)
    rows = []
    for frame, boxes in grouped.items():
        if start <= frame <= end:
            for x1, y1, x2, y2, conf, class_id in boxes:
                rows.append((frame - offset, x1, y1, x2, y2, conf, class_id))
    rows.sort(key=lambda row: (row[0], -row[5]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(DETECTION_COLUMNS)
        writer.writerows(rows)
    return dest


def mot_to_detections(path, keep=None):
    grouped = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            values = line.strip().split(",")
            frame = int(float(values[0]))
            left, top, width, height = map(float, values[2:6])
            if keep is not None and not keep(values):
                continue
            conf = float(values[6]) if len(values) > 6 else 1.0
            grouped[frame].append([left, top, left + width, top + height, conf, 0.0])
    return grouped


def gold_keep(values):
    return int(float(values[6])) == 1 and int(float(values[7])) == 1


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        out = [0] * len(values)
        for rank, index in enumerate(order, start=1):
            out[index] = rank
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return None
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else None


def combine_rows(rows, extra):
    grouped = defaultdict(list)
    raws = defaultdict(dict)
    for row, raw, sequence_name in extra:
        key = (row["detector"], row["tracker"])
        grouped[key].append(row)
        raws[key][sequence_name] = raw
    summary = []
    for (detector, tracker), items in grouped.items():
        combined = combine_trackeval(raws[(detector, tracker)])
        frames = int(sum(row["frames"] for row in items))

        def weighted(key):
            return 100.0 * sum(row[key] / 100.0 * row["frames"] for row in items) / frames

        item = {
            "detector": detector,
            "tracker": tracker,
            "sequences": len(items),
            "frames": frames,
            **combined,
            "HOTA_unweighted_mean": mean([row["HOTA"] for row in items]),
            "assertion_pass_rate": 100.0 * (
                sum(row["frames"] - row["assertions_failed"] for row in items) / frames
            ),
            "A1_pass_rate": weighted("A1_pass_rate"),
            "A2_pass_rate": weighted("A2_pass_rate"),
            "A3_pass_rate": weighted("A3_pass_rate"),
            "A4_pass_rate": weighted("A4_pass_rate"),
            "m1_motion_pass_rate": mean([row["m1_motion_pass_rate"] for row in items]),
            "IDSW": int(sum(row["IDSW"] for row in items)),
            "det_precision": None,
            "det_recall": None,
        }
        if all("det_tp" in row for row in items):
            tp = sum(row["det_tp"] for row in items)
            fp = sum(row["det_fp"] for row in items)
            fn = sum(row["det_fn"] for row in items)
            item["det_precision"] = 100.0 * tp / (tp + fp) if (tp + fp) else None
            item["det_recall"] = 100.0 * tp / (tp + fn) if (tp + fn) else None
        summary.append(item)
    summary.sort(key=lambda row: (-(row["HOTA"] or 0), -(row["IDF1"] or 0)))
    return summary


def evaluate_one(sequence, detections, tracker_name, config, tracks_path, detector, tracker_tag):
    run_tracking(sequence, detections, tracker_name, config, tracks_path)
    result, raw = score_tracks_with_raw(sequence, tracks_path, detections=detections)
    result["detector"] = detector
    result["tracker"] = tracker_tag
    print(
        f"  {sequence.name} {detector}+{tracker_tag}: "
        f"HOTA {result['HOTA']:.1f} MOTA {result['MOTA']:.1f} IDF1 {result['IDF1']:.1f} "
        f"IDSW {result['IDSW']}  A1–A3 {result['assertion_pass_rate']:.1f}%  "
        f"M1-motion {result['m1_motion_pass_rate']:.1f}%",
        flush=True,
    )
    return result, raw


def main():
    args = parse_args()
    which = args.slice
    manifest = load_manifest()
    dataset_root = slice_dataset_root(which)
    sequences = [load_sequence(dataset_root, name) for name in GUIDE_SEQUENCES]
    out = args.output_root / which
    per_seq = []
    extras = []

    print(f"slice={which}  sequences={[s.name for s in sequences]}", flush=True)

    for sequence in sequences:
        start, end = manifest[which][sequence.name]["original_frames"]
        dpm = DPM_NAME[sequence.name]
        for tag in M1_DETECTORS:
            src = args.full_detections / dpm / f"{tag}.csv"
            dest = out / "detections" / sequence.name / f"{tag}.csv"
            if not src.is_file():
                raise FileNotFoundError(src)
            remap_full_detections(src, dest, start, end)
            detections = read_detections(dest)
            for tracker_name, config, tracker_tag in TRACKERS:
                tracks_path = out / "tracks" / sequence.name / f"{tag}_{tracker_tag}.txt"
                result, raw = evaluate_one(
                    sequence, detections, tracker_name, config, tracks_path, tag, tracker_tag)
                per_seq.append(result)
                extras.append((result, raw, sequence.name))

        public = mot_to_detections(sequence.directory / "det" / "det.txt")
        write_detections(out / "detections" / sequence.name / "public_frcnn.csv",
                         [(frame, *box[:5]) for frame, boxes in public.items() for box in boxes])
        result, raw = evaluate_one(
            sequence, public, "bytetrack", PIPELINE / "configs" / "bytetrack.yaml",
            out / "tracks" / sequence.name / "public_frcnn_bytetrack.txt",
            "public_frcnn", "bytetrack")
        per_seq.append(result)
        extras.append((result, raw, sequence.name))

        gold = mot_to_detections(sequence.directory / "gt" / "gt.txt", keep=gold_keep)
        write_detections(out / "detections" / sequence.name / "gold_gt.csv",
                         [(frame, *box[:5]) for frame, boxes in gold.items() for box in boxes])
        result, raw = evaluate_one(
            sequence, gold, "bytetrack", PIPELINE / "configs" / "bytetrack.yaml",
            out / "tracks" / sequence.name / "gold_gt_bytetrack.txt",
            "gold_gt", "bytetrack")
        per_seq.append(result)
        extras.append((result, raw, sequence.name))

    summary = combine_rows(per_seq, extras)
    write_csv(args.results_dir / f"{which}_per_sequence.csv", per_seq)
    write_csv(args.results_dir / f"{which}_summary.csv", summary)

    m1_rows = [row for row in summary if row["detector"] in M1_DETECTORS]
    hota = [row["HOTA"] for row in m1_rows]
    asserts = [row["assertion_pass_rate"] for row in m1_rows]
    m1_motion = [row["m1_motion_pass_rate"] for row in m1_rows]
    hota_order = [row["detector"] + "+" + row["tracker"] for row in sorted(m1_rows, key=lambda r: -r["HOTA"])]
    assert_order = [row["detector"] + "+" + row["tracker"] for row in sorted(m1_rows, key=lambda r: -r["assertion_pass_rate"])]
    report = {
        "slice": which,
        "guide": "Pipeline Evaluation Guide v4",
        "primary_metric": "TrackEval HOTA (MOT17 pedestrian preprocessing), combined across sequences",
        "also": ["MOTA", "IDF1", "IDSW", "detector precision/recall at IoU 0.5"],
        "baseline": "ByteTrack defaults on MOT17 public FRCNN detections",
        "gold_input": "ByteTrack defaults on GT pedestrian boxes (class 1, consider-flag 1)",
        "required_assertions": {
            "A1": "track centre jump ≤ 0.5 × current box height (tracks present in both frames)",
            "A2": "tracked box height/width in [1.0, 6.0]",
            "A3": "|count_t - count_{t-1}| ≤ max(3, 0.3 × previous count)",
            "pass": "frame passes iff A1 ∧ A2 ∧ A3",
        },
        "extra_assertions": {
            "A4": "every tracked box overlaps the image rectangle (online; drifted off-screen tracks are not pedestrians)",
            "m1_motion": "Milestone 1: centre jump / image diagonal / frame ≤ 0.0157 (tuned on all 7 train videos, including this test slice — leaked)",
        },
        "milestone1_vs_guide": {
            "m1_accuracy": "unweighted mean HOTA over 7 full DPM sequences",
            "guide_accuracy": "TrackEval-combined HOTA on second halves of 02/04/09 FRCNN",
            "m1_assertion": "per track-step diagonal-normalized speed",
            "guide_assertion": "per-frame A1∧A2∧A3",
        },
        "ranking": {
            "n_m1_configs": len(m1_rows),
            "spearman_hota_vs_A1A2A3": spearman(hota, asserts),
            "pearson_hota_vs_A1A2A3": pearson(hota, asserts),
            "spearman_hota_vs_m1_motion": spearman(hota, m1_motion),
            "pearson_hota_vs_m1_motion": pearson(hota, m1_motion),
            "same_ranking_as_hota": hota_order == assert_order,
            "hota_order": hota_order,
            "assertion_order": assert_order,
        },
        "combined": summary,
    }
    (args.results_dir / f"{which}_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\ncombined ({which}):")
    for row in summary:
        print(f"  {row['detector']:<22} {row['tracker']:<14} HOTA {row['HOTA']:.1f}  "
              f"MOTA {row['MOTA']:.1f}  IDF1 {row['IDF1']:.1f}  IDSW {row['IDSW']:<5}  "
              f"A1–A3 {row['assertion_pass_rate']:.1f}%  M1 {row['m1_motion_pass_rate']:.1f}%")
    print(f"\nSpearman HOTA vs A1–A3: {report['ranking']['spearman_hota_vs_A1A2A3']}")
    print(f"same ranking: {report['ranking']['same_ranking_as_hota']}")
    print(f"-> {args.results_dir / f'{which}_summary.csv'}")


if __name__ == "__main__":
    main()
