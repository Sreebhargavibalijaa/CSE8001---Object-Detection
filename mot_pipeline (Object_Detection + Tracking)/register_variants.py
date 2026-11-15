"""Export, score and register at least four variants per scalable stage.

Scalable stages in this pipeline:
  detection — the person detector (family, input size, frame skip, ONNX / INT8)
  tracking  — ByteTrack / OC-SORT configs (thresholds, lost-track buffer)

Offline accuracy is TrackEval on a held-out test slice: MOT17-02-DPM, frames 1-100.
Detection variants are all tracked with default ByteTrack; tracking variants all
reuse YOLO26n imgsz=640 detections so the tracker is the only change.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import REPO_ROOT, load_sequence, read_detections, resolve_device
from detect import build_detector, run_detection
from evaluate import score_tracks
from registry import ModelRegistry, metadata, print_catalog
from track import run_tracking

PIPELINE_DIR = Path(__file__).resolve().parent
ARTIFACTS = PIPELINE_DIR / "registry" / "export"
VERSION = "1.0.0"
TEST_SEQUENCE = "MOT17-02-DPM"
TEST_FRAMES = 100
ACCURACY_KEYS = ("HOTA", "MOTA", "IDF1", "Precision", "Recall", "assertion_pass_rate")

DETECTION_VARIANTS = (
    {
        "name": "yolo26n-pt-imgsz640",
        "source": "Ultralytics YOLO26n PyTorch weights, imgsz=640",
        "detector": "yolo26n", "imgsz": 640, "frame_skip": 1, "runtime": "pytorch",
        "kind": "family+default",
    },
    {
        "name": "yolo26n-pt-imgsz1280",
        "source": "Ultralytics YOLO26n PyTorch weights, imgsz=1280 (input resolution)",
        "detector": "yolo26n", "imgsz": 1280, "frame_skip": 1, "runtime": "pytorch",
        "kind": "input-resolution",
    },
    {
        "name": "yolo26n-pt-imgsz640-skip2",
        "source": "Ultralytics YOLO26n PyTorch, imgsz=640, detect every 2nd frame",
        "detector": "yolo26n", "imgsz": 640, "frame_skip": 2, "runtime": "pytorch",
        "kind": "frame-skip",
    },
    {
        "name": "yolo26n-onnx-imgsz640",
        "source": "YOLO26n exported to ONNX (graph optimization, CPU Runtime)",
        "detector": "yolo26n", "imgsz": 640, "frame_skip": 1, "runtime": "onnx",
        "kind": "onnx",
    },
    {
        "name": "yolo26n-onnx-int8-imgsz640",
        "source": "YOLO26n ONNX with ONNX Runtime dynamic INT8 quantization",
        "detector": "yolo26n", "imgsz": 640, "frame_skip": 1, "runtime": "onnx-int8",
        "kind": "onnx-int8",
    },
)

TRACKING_VARIANTS = (
    {
        "name": "bytetrack",
        "source": "Ultralytics ByteTrack defaults (track_high_thresh=0.25, buffer=30)",
        "tracker": "bytetrack",
        "config": PIPELINE_DIR / "configs" / "bytetrack.yaml",
    },
    {
        "name": "bytetrack-low",
        "source": "ByteTrack with lower association thresholds (keep weaker detections)",
        "tracker": "bytetrack",
        "config": PIPELINE_DIR / "configs" / "bytetrack_low.yaml",
    },
    {
        "name": "bytetrack-long",
        "source": "ByteTrack with track_buffer=90 (longer occlusions)",
        "tracker": "bytetrack",
        "config": PIPELINE_DIR / "configs" / "bytetrack_long.yaml",
    },
    {
        "name": "ocsort",
        "source": "Ultralytics OC-SORT defaults",
        "tracker": "ocsort",
        "config": PIPELINE_DIR / "configs" / "ocsort.yaml",
    },
)


def slim_accuracy(result):
    return {key: (None if result.get(key) is None else round(float(result[key]), 3))
            for key in ACCURACY_KEYS} | {
        "test_slice": f"{TEST_SEQUENCE} frames 1-{TEST_FRAMES}",
        "FP": result.get("FP"), "FN": result.get("FN"), "IDSW": result.get("IDSW"),
    }


def export_yolo_onnx(pt_path, onnx_path, imgsz):
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    produced = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != onnx_path.resolve():
        shutil.copy2(produced, onnx_path)
        if produced.parent == pt_path.parent and produced.name.endswith(".onnx"):
            pass
    print(f"exported ONNX -> {onnx_path} ({onnx_path.stat().st_size} bytes)", flush=True)
    return onnx_path


def export_yolo_int8(onnx_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    # Quantize MatMul/Gemm only. Full ConvInteger graphs are not implemented
    # on CoreML and some CPU ORT builds used with Ultralytics.
    quantize_dynamic(model_input=str(onnx_path), model_output=str(int8_path),
                     weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    print(f"quantized INT8 -> {int8_path} ({int8_path.stat().st_size} bytes)", flush=True)
    return int8_path


def prepare_artifacts(pt_path, skip_export=False):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    copied = ARTIFACTS / "yolo26n.pt"
    if Path(pt_path).resolve() != copied.resolve():
        shutil.copy2(pt_path, copied)
    onnx_path = ARTIFACTS / "yolo26n.onnx"
    int8_path = ARTIFACTS / "yolo26n.int8.onnx"
    if not skip_export:
        if not onnx_path.is_file():
            export_yolo_onnx(copied, onnx_path, imgsz=640)
        if onnx_path.is_file() and not int8_path.is_file():
            try:
                export_yolo_int8(onnx_path, int8_path)
            except Exception as error:
                print(f"INT8 quantization failed ({error}); skipping that variant", flush=True)
    available = {"pytorch": copied}
    if onnx_path.is_file():
        available["onnx"] = onnx_path
    if int8_path.is_file():
        available["onnx-int8"] = int8_path
    return available


def score_detection_variant(args, sequence, variant, weights):
    device = resolve_device(args.device)
    tag = variant["name"]
    detect_fn = build_detector(
        variant["detector"], args.conf, device, variant["imgsz"], weights,
        frame_skip=variant["frame_skip"], runtime=variant["runtime"], tag=tag)[1]
    det_path = args.output_root / "detections" / sequence.name / f"{tag}.csv"
    fps, n_boxes, n_frames = run_detection(
        sequence, detect_fn, det_path, args.max_frames, tag, variant["frame_skip"])
    tracks_path = args.output_root / "tracks" / sequence.name / f"{tag}_bytetrack.txt"
    run_tracking(sequence, read_detections(det_path), "bytetrack",
                 PIPELINE_DIR / "configs" / "bytetrack.yaml", tracks_path, args.max_frames)
    result = score_tracks(sequence, tracks_path, args.max_frames)
    result["fps"] = fps
    print(f"  {tag}: HOTA {result['HOTA']:.1f}  Rec {result['Recall']:.1f}  "
          f"{fps:.2f} FPS  ({n_boxes} boxes / {n_frames} frames)", flush=True)
    return result, fps


def score_tracking_variant(args, sequence, variant, detections):
    tag = variant["name"]
    tracks_path = args.output_root / "tracks" / sequence.name / f"yolo26n_imgsz640_{tag}.txt"
    run_tracking(sequence, detections, variant["tracker"], variant["config"],
                 tracks_path, args.max_frames)
    result = score_tracks(sequence, tracks_path, args.max_frames)
    print(f"  {tag}: HOTA {result['HOTA']:.1f}  Rec {result['Recall']:.1f}  "
          f"IDF1 {result['IDF1']:.1f}", flush=True)
    return result


def register_detection(registry, variant, artifact, accuracy, extra):
    record = metadata(
        name=variant["name"], stage="detection", version=VERSION,
        source=variant["source"], artifact=artifact, extra=extra, accuracy=accuracy)
    registry.put_file("detection", variant["name"], VERSION, artifact)
    registry.put_metadata(record)
    return record


def register_tracking(registry, variant, accuracy):
    record = metadata(
        name=variant["name"], stage="tracking", version=VERSION,
        source=variant["source"], artifact=variant["config"],
        extra={"tracker": variant["tracker"], "config": variant["config"].name},
        accuracy=accuracy)
    registry.put_file("tracking", variant["name"], VERSION, variant["config"])
    registry.put_metadata(record)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Export, evaluate and register pipeline variants")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "MOT17")
    parser.add_argument("--output-root", type=Path, default=PIPELINE_DIR / "outputs")
    parser.add_argument("--yolo-weights", type=Path, default=REPO_ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    args.max_frames = TEST_FRAMES
    return args


def main():
    args = parse_args()
    sequence = load_sequence(args.dataset_root, TEST_SEQUENCE)
    print(f"test slice: {TEST_SEQUENCE} frames 1-{TEST_FRAMES}", flush=True)

    artifacts = prepare_artifacts(args.yolo_weights, skip_export=args.skip_export)
    registry = ModelRegistry()
    records = []

    print("\n=== detection variants ===", flush=True)
    for variant in DETECTION_VARIANTS:
        if variant["runtime"] not in artifacts:
            print(f"skip {variant['name']}: {variant['runtime']} artifact missing", flush=True)
            continue
        weights = artifacts[variant["runtime"]]
        accuracy = fps = None
        if not args.skip_eval:
            try:
                result, fps = score_detection_variant(args, sequence, variant, weights)
                accuracy = slim_accuracy(result)
            except Exception as error:
                print(f"skip {variant['name']}: eval failed ({error})", flush=True)
                continue
        extra = {
            "runtime": variant["runtime"],
            "imgsz": variant["imgsz"],
            "frame_skip": variant["frame_skip"],
            "kind": variant["kind"],
        }
        if fps is not None:
            extra["test_slice_fps"] = round(fps, 3)
        records.append(register_detection(registry, variant, weights, accuracy, extra))

    print("\n=== tracking variants ===", flush=True)
    baseline_det = args.output_root / "detections" / sequence.name / "yolo26n-pt-imgsz640.csv"
    if not baseline_det.is_file():
        baseline_det = args.output_root / "detections" / sequence.name / "yolo26n_imgsz640.csv"
    detections = read_detections(baseline_det)
    for variant in TRACKING_VARIANTS:
        accuracy = None
        if not args.skip_eval:
            result = score_tracking_variant(args, sequence, variant, detections)
            accuracy = slim_accuracy(result)
        records.append(register_tracking(registry, variant, accuracy))

    print("\nRegistered variants:")
    print_catalog(registry.list_models())
    print(f"\nLocal catalog -> {PIPELINE_DIR / 'registry' / 'catalog.json'}")


if __name__ == "__main__":
    main()
