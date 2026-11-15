"""Accuracy is a function of variant, not replicas. Look up Guide v4 test-slice HOTA."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
GUIDE = PIPELINE / "results" / "guide_eval" / "test_summary.csv"
CATALOG = PIPELINE / "registry" / "catalog.json"


def _guide_rows():
    with GUIDE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def detect_hota(variant, tracker="bytetrack"):
    key = {
        "yolo26n_imgsz640": "yolo26n_imgsz640",
        "yolo26n_imgsz1280": "yolo26n_imgsz1280",
        "yolo26n-onnx-imgsz640": None,
        "yolo26n-onnx-int8-imgsz640": None,
    }[variant]
    if key:
        for row in _guide_rows():
            if row["detector"] == key and row["tracker"] == tracker:
                return float(row["HOTA"]), "guide test slice (02/04/09 FRCNN second half) + ByteTrack"
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    name = variant
    for model in catalog["models"]:
        if model["stage"] == "detection" and model["name"] == name:
            acc = model.get("offline_accuracy") or {}
            return float(acc["HOTA"]), "T2 slice MOT17-02-DPM frames 1–100 (ONNX not in 3-seq test eval)"
    raise KeyError(variant)


def track_hota(variant, detector="yolo26n_imgsz640"):
    key = {
        "bytetrack": "bytetrack",
        "bytetrack_low": "bytetrack_low",
        "bytetrack_long": "bytetrack",
        "ocsort": "ocsort",
    }[variant]
    for row in _guide_rows():
        if row["detector"] == detector and row["tracker"] == key:
            note = "guide test slice; bytetrack_long uses default ByteTrack HOTA (same detections)"
            if variant == "bytetrack_long":
                note += " (buffer=90; HOTA matched default on this slice)"
            return float(row["HOTA"]), note
    raise KeyError(variant)


def e2e_hota(detect, track):
    track_key = {"bytetrack_long": "bytetrack"}.get(track, track)
    detect_key = detect
    if detect.startswith("yolo26n-onnx"):
        hota, note = detect_hota(detect)
        return hota, note + f"; tracker={track}"
    if detect == "gold_gt":
        for row in _guide_rows():
            if row["detector"] == "gold_gt":
                return float(row["HOTA"]), (
                    "gold-input ablation on test slice (GT pedestrian boxes → tracker); "
                    f"scored with {row['tracker']}"
                )
    for row in _guide_rows():
        if row["detector"] == detect_key and row["tracker"] == track_key:
            return float(row["HOTA"]), "guide test slice combined HOTA"
    raise KeyError((detect, track))
