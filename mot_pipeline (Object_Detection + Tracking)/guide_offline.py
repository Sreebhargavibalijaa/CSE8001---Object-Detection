"""Guide v4 offline accuracy for every registered variant (test slice only).

For each registered detection variant this runs the detector once over the test
slice (second halves of MOT17-02/04/09-FRCNN, renumbered from 1) and saves the
boxes. Every tracking variant is then run on every detector's boxes, plus the
two reference rows the guide asks for:

  baseline   ByteTrack (default) on the MOT17 public FRCNN detections
  gold input each tracker on ground-truth pedestrian boxes

Outputs (results/guide_eval_vm/):
  test_per_sequence.csv   one row per sequence x detector x tracker
  test_summary.csv        TrackEval-combined rows (HOTA, MOTA, IDF1, IDSW, ...)
  test_report.json        assertion vs HOTA ranking, settings, slice definition

The saved test-slice detections (outputs/guide/test/detections/<seq>/<variant>.csv)
are the cached upstream outputs that the k8s tracker profiling reuses.
Registry metadata offline_accuracy is overwritten with these test-slice numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_sequence, read_detections, write_detections
from detect import build_detector, run_detection
from eval_harness import (
    combine_rows, evaluate_one, gold_keep, mot_to_detections, pearson, spearman, write_csv,
)
from registry import ModelRegistry
from slices import GUIDE_SEQUENCES, slice_dataset_root

PIPELINE = Path(__file__).resolve().parent
VERSION = "1.0.0"
REFERENCE_DETECTOR = "yolo26n-pt-imgsz640"
REFERENCE_TRACKER = "bytetrack"
SLICE_NOTE = "Guide v4 test slice: second halves of MOT17-02/04/09-FRCNN (1088 frames)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=PIPELINE / "results" / "guide_eval_vm")
    parser.add_argument("--output-root", type=Path, default=PIPELINE / "outputs" / "guide" / "test")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="re-run detection even if CSVs exist")
    parser.add_argument("--detectors", nargs="*", help="only these detection variants")
    return parser.parse_args()


def artifact_path(stage, entry):
    return (PIPELINE / "registry" / "artifacts" / stage / entry["name"] / entry["version"]
            / entry["artifact"])


def detection_variants(registry, only=None):
    out = []
    for entry in registry.list_models():
        if entry["stage"] != "detection" or entry["version"] != VERSION:
            continue
        if only and entry["name"] not in only:
            continue
        runtime = entry.get("runtime", "pytorch")
        out.append({
            "name": entry["name"],
            "detector": entry.get("detector") or ("fasterrcnn" if runtime == "torchvision" else "yolo26n"),
            "runtime": runtime,
            "imgsz": int(entry.get("imgsz") or 640),
            "frame_skip": int(entry.get("frame_skip") or 1),
            "weights": artifact_path("detection", entry),
        })
    return out


def tracking_variants(registry):
    out = []
    for entry in registry.list_models():
        if entry["stage"] != "tracking" or entry["version"] != VERSION:
            continue
        out.append({
            "name": entry["name"],
            "tracker": entry.get("tracker", "bytetrack"),
            "config": PIPELINE / "configs" / entry.get("config", f"{entry['name']}.yaml"),
        })
    return out


def artifact_stamp(variant):
    weights = Path(variant["weights"])
    size = weights.stat().st_size if weights.is_file() else None
    return {"artifact": weights.name, "size": size, "conf": None}


def detect_slice(variant, sequences, out_root, conf, force):
    stamp = artifact_stamp(variant) | {"conf": conf}
    for sequence in sequences:
        dest = out_root / "detections" / sequence.name / f"{variant['name']}.csv"
        sidecar = dest.with_suffix(".source.json")
        cached = sidecar.is_file() and json.loads(sidecar.read_text(encoding="utf-8")) == stamp
        if dest.is_file() and cached and not force:
            print(f"  cached {dest.relative_to(PIPELINE)}", flush=True)
            continue
        _, detect = build_detector(
            variant["detector"], conf, "cpu", variant["imgsz"], variant["weights"],
            frame_skip=variant["frame_skip"], runtime=variant["runtime"], tag=variant["name"])
        run_detection(sequence, detect, dest, None, variant["name"], variant["frame_skip"])
        sidecar.write_text(json.dumps(stamp), encoding="utf-8")


def slim(row):
    keys = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Precision", "Recall",
            "det_precision", "det_recall", "assertion_pass_rate",
            "A1_pass_rate", "A2_pass_rate", "A3_pass_rate", "A4_pass_rate")
    out = {}
    for key in keys:
        value = row.get(key)
        out[key] = None if value is None else round(float(value), 3)
    out["test_slice"] = SLICE_NOTE
    return out


def update_registry(registry, summary):
    lookup = {(row["detector"], row["tracker"]): row for row in summary}
    for entry in registry.list_models():
        if entry["version"] != VERSION:
            continue
        if entry["stage"] == "detection":
            row = lookup.get((entry["name"], REFERENCE_TRACKER))
            paired = f"tracked with {REFERENCE_TRACKER}"
        else:
            row = lookup.get((REFERENCE_DETECTOR, entry["name"]))
            paired = f"on {REFERENCE_DETECTOR} detections"
        if row is None:
            continue
        meta = registry.get_metadata(entry["stage"], entry["name"], entry["version"])
        meta["offline_accuracy"] = slim(row) | {"paired_with": paired}
        registry.put_metadata(meta)
        print(f"  registry {entry['stage']}/{entry['name']}: HOTA {row['HOTA']:.1f} ({paired})",
              flush=True)


def main():
    args = parse_args()
    root = slice_dataset_root("test")
    sequences = [load_sequence(root, name) for name in GUIDE_SEQUENCES]
    registry = ModelRegistry()
    detectors = detection_variants(registry, args.detectors)
    trackers = tracking_variants(registry)
    print(f"test slice: {[s.name for s in sequences]}  "
          f"({sum(s.seq_length for s in sequences)} frames)", flush=True)
    print(f"detectors: {[d['name'] for d in detectors]}", flush=True)
    print(f"trackers:  {[t['name'] for t in trackers]}", flush=True)

    out = args.output_root
    per_seq, extras = [], []
    for variant in detectors:
        print(f"\n=== detect {variant['name']} ===", flush=True)
        detect_slice(variant, sequences, out, args.conf, args.force)

    for sequence in sequences:
        for variant in detectors:
            detections = read_detections(out / "detections" / sequence.name / f"{variant['name']}.csv")
            for tracker in trackers:
                tracks = out / "tracks" / sequence.name / f"{variant['name']}_{tracker['name']}.txt"
                result, raw = evaluate_one(sequence, detections, tracker["tracker"], tracker["config"],
                                           tracks, variant["name"], tracker["name"])
                per_seq.append(result)
                extras.append((result, raw, sequence.name))

        public = mot_to_detections(sequence.directory / "det" / "det.txt")
        write_detections(out / "detections" / sequence.name / "public_frcnn.csv",
                         [(frame, *box[:5]) for frame, boxes in public.items() for box in boxes])
        result, raw = evaluate_one(sequence, public, "bytetrack", PIPELINE / "configs" / "bytetrack.yaml",
                                   out / "tracks" / sequence.name / "public_frcnn_bytetrack.txt",
                                   "public_frcnn", "bytetrack")
        per_seq.append(result)
        extras.append((result, raw, sequence.name))

        gold = mot_to_detections(sequence.directory / "gt" / "gt.txt", keep=gold_keep)
        write_detections(out / "detections" / sequence.name / "gold_gt.csv",
                         [(frame, *box[:5]) for frame, boxes in gold.items() for box in boxes])
        for tracker in trackers:
            result, raw = evaluate_one(sequence, gold, tracker["tracker"], tracker["config"],
                                       out / "tracks" / sequence.name / f"gold_gt_{tracker['name']}.txt",
                                       "gold_gt", tracker["name"])
            per_seq.append(result)
            extras.append((result, raw, sequence.name))

    summary = combine_rows(per_seq, extras)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "test_per_sequence.csv", per_seq)
    write_csv(args.results_dir / "test_summary.csv", summary)

    deployable = [row for row in summary if row["detector"] not in ("gold_gt", "public_frcnn")]
    hota = [row["HOTA"] for row in deployable]
    passes = [row["assertion_pass_rate"] for row in deployable]
    key = lambda row: f"{row['detector']}+{row['tracker']}"  # noqa: E731
    report = {
        "slice": SLICE_NOTE,
        "primary_metric": "TrackEval HOTA (MOT17 pedestrian settings), combined over the 3 sequences",
        "baseline": "ByteTrack defaults on MOT17 public FRCNN detections",
        "gold_input": "each tracker on GT pedestrian boxes (class 1, consider-flag 1)",
        "detector_conf_floor": args.conf,
        "n_configs": len(deployable),
        "spearman_hota_vs_assertion_pass": spearman(hota, passes),
        "pearson_hota_vs_assertion_pass": pearson(hota, passes),
        "hota_order": [key(r) for r in sorted(deployable, key=lambda r: -r["HOTA"])],
        "assertion_order": [key(r) for r in sorted(deployable, key=lambda r: -r["assertion_pass_rate"])],
        "combined": summary,
    }
    (args.results_dir / "test_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\nupdating registry offline_accuracy with test-slice numbers", flush=True)
    update_registry(registry, summary)

    print("\ncombined (test slice):")
    for row in summary:
        print(f"  {row['detector']:<28} {row['tracker']:<15} HOTA {row['HOTA']:5.1f}  "
              f"MOTA {row['MOTA']:5.1f}  IDF1 {row['IDF1']:5.1f}  IDSW {row['IDSW']:<5}  "
              f"checks {row['assertion_pass_rate']:5.1f}%")
    print(f"\nSpearman HOTA vs check pass rate: {report['spearman_hota_vs_assertion_pass']}")
    print(f"-> {args.results_dir / 'test_summary.csv'}")


if __name__ == "__main__":
    main()
