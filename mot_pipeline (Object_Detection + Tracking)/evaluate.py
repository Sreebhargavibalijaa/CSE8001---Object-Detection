"""Step 3: score a MOT track file against the MOT17 ground truth.

Official TrackEval MOT17 preprocessing with the HOTA, CLEAR and Identity metric families,
plus a motion assertion that flags physically implausible jumps within a track.
Writes <output-root>/metrics/<sequence>/<track file name>.csv
"""

import argparse
import csv
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trackeval

from common import add_common_arguments, load_sequence, read_tracks

# 99th percentile of ground-truth pedestrian centre movement, in image diagonals per frame,
# calibrated on the seven MOT17 training videos (111,751 samples).
MOTION_THRESHOLD = 0.01570218735615795


def tracking_metrics(sequence, tracks_path, max_frames=None):
    """Percentages except FP, FN, IDSW and Frag, which are counts. MOTP is mean matched IoU."""
    limit = sequence.frame_limit(max_frames)
    with tempfile.TemporaryDirectory(prefix="trackeval-") as stage:
        # TrackEval expects <tracker>/data/<sequence>.txt; GT is read from the dataset itself.
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
        # Load the full ground truth, then keep only the evaluated time window.
        for key, value in raw.items():
            if isinstance(value, list):
                raw[key] = value[:limit]
        raw["num_timesteps"] = limit
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
    hota, clear, identity = (getattr(trackeval.metrics, name)({"PRINT_CONFIG": False}).eval_sequence(data)
                             for name in ("HOTA", "CLEAR", "Identity"))
    metrics = {key: float(np.mean(hota[key]) * 100) for key in ("HOTA", "DetA", "AssA", "LocA")}
    metrics.update({key: float(clear[key] * 100) for key in ("MOTA", "MOTP")})
    metrics.update({key: float(identity[key] * 100) for key in ("IDF1", "IDP", "IDR")})
    metrics.update({"FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"]), "IDSW": int(clear["IDSW"]),
                    "Frag": int(clear["Frag"]), "Precision": float(clear["CLR_Pr"] * 100),
                    "Recall": float(clear["CLR_Re"] * 100)})
    return metrics


def motion_assertion(sequence, tracks, threshold):
    """Centre displacement / image diagonal / frame gap must stay within the threshold."""
    centres = defaultdict(list)
    for frame, track_id, left, top, width, height, _ in tracks:
        centres[track_id].append((frame, left + width / 2, top + height / 2))
    diagonal = math.hypot(sequence.width, sequence.height)
    checked = passed = 0
    for observations in centres.values():
        ordered = sorted(observations)
        for first, second in zip(ordered, ordered[1:]):
            speed = math.hypot(second[1] - first[1], second[2] - first[2]) / diagonal / (second[0] - first[0])
            checked += 1
            passed += speed <= threshold
    return {"assertions_checked": checked, "assertions_failed": checked - passed,
            "assertion_pass_rate": 100.0 * passed / checked if checked else None}


def main():
    parser = argparse.ArgumentParser(description="TrackEval metrics and motion assertion for one track file")
    add_common_arguments(parser)
    parser.add_argument("--tracks", type=Path, required=True, help="MOT file written by track.py")
    parser.add_argument("--motion-threshold", type=float, default=MOTION_THRESHOLD)
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    limit = sequence.frame_limit(args.max_frames)
    tracks = [row for row in read_tracks(args.tracks) if row[0] <= limit]
    result = {"sequence": sequence.name, "tracks": args.tracks.stem, "frames": limit,
              **tracking_metrics(sequence, args.tracks, args.max_frames),
              **motion_assertion(sequence, tracks, args.motion_threshold)}

    for key, value in result.items():
        print(f"{key:>22}: {value:.3f}" if isinstance(value, float) else f"{key:>22}: {value}")
    output = args.output_root / "metrics" / sequence.name / f"{args.tracks.stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
