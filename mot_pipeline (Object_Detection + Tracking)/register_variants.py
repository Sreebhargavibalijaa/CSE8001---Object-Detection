"""Export, score and register at least four variants per scalable stage.

Scalable stages in this pipeline:
  detection — the person detector (family, input size, frame skip, ONNX / INT8)
  tracking  — ByteTrack / OC-SORT configs (thresholds, lost-track buffer)

A quick smoke score is computed on MOT17-02-DPM frames 1-100 while registering.
guide_offline.py then overwrites offline_accuracy with Guide v4 test-slice numbers
(second halves of MOT17-02/04/09-FRCNN). Detection variants are tracked with default
ByteTrack; tracking variants reuse YOLO26n imgsz=640 detections.
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
        "source": ("YOLO26n ONNX, ONNX Runtime static INT8 (QDQ, Conv layers, "
                   "calibrated on 32 development-slice frames)"),
        "detector": "yolo26n", "imgsz": 640, "frame_skip": 1, "runtime": "onnx-int8",
        "kind": "onnx-int8",
    },
    {
        "name": "fasterrcnn-mv3-320",
        "source": "torchvision Faster R-CNN MobileNetV3-Large 320-FPN, COCO weights (other family)",
        "detector": "fasterrcnn", "imgsz": 320, "frame_skip": 1, "runtime": "torchvision",
        "kind": "family",
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


def calibration_frames(count=32, seed=8001):
    """Development-slice frames for INT8 calibration (never the test slice)."""
    import numpy as np
    from slices import GUIDE_SEQUENCES, slice_dataset_root
    root = slice_dataset_root("development")
    paths = []
    for name in GUIDE_SEQUENCES:
        folder = root / "train" / name / "img1"
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} missing; run build_slices.py first")
        paths.extend(sorted(folder.glob("*.jpg")))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(paths), size=min(count, len(paths)), replace=False)
    return [paths[i] for i in sorted(picks)]


def letterbox_tensor(path, size=640):
    """Same preprocessing as Ultralytics: keep aspect, pad with 114, RGB, 0..1, NCHW."""
    import cv2
    import numpy as np
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = image
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor)


def export_yolo_int8(onnx_path, int8_path, imgsz=640):
    """Static INT8 for CPU: quantize Conv layers (QDQ), calibrated on dev-slice frames.

    The Milestone 1 dynamic version only touched MatMul/Gemm, which YOLO barely
    uses, so it was the same model as FP32 ONNX. On a Linux x86 CPU, ONNX Runtime
    runs QDQ Conv as real INT8 kernels. Post-processing stays in float.
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    source = onnx.load(str(onnx_path))
    input_name = source.graph.input[0].name
    frames = calibration_frames()

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter([{input_name: letterbox_tensor(path, imgsz)} for path in frames])

        def get_next(self):
            return next(self.items, None)

    prepared = int8_path.with_name(int8_path.stem + ".prep.onnx")
    upgraded = int8_path.with_name(int8_path.stem + ".opset13.onnx")
    model_input = onnx_path
    # Per-channel QDQ needs DequantizeLinear(axis), i.e. opset >= 13; the
    # Milestone 1 export used opset 12. Upgrade a copy; fall back to per-tensor.
    per_channel = False
    opset = max((entry.version for entry in source.opset_import if entry.domain in ("", "ai.onnx")),
                default=0)
    if opset >= 13:
        per_channel = True
    else:
        try:
            from onnx import version_converter
            onnx.save(version_converter.convert_version(source, 13), str(upgraded))
            model_input, per_channel = upgraded, True
        except Exception as error:
            print(f"opset 13 upgrade failed ({error}); using per-tensor INT8", flush=True)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(model_input), str(prepared))
        model_input = prepared
    except Exception as error:
        print(f"quant_pre_process skipped ({error})", flush=True)

    def quantize(per_channel_flag):
        quantize_static(str(model_input), str(int8_path), Reader(),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=per_channel_flag, op_types_to_quantize=["Conv"])

    quantize(per_channel)
    if not onnx_loads(int8_path, input_name, frames[0], imgsz) and per_channel:
        print("per-channel INT8 did not load; retrying per-tensor", flush=True)
        quantize(False)
        per_channel = False
    # Ultralytics reads class names / stride / imgsz from the ONNX metadata.
    quantized = onnx.load(str(int8_path))
    have = {entry.key for entry in quantized.metadata_props}
    for entry in source.metadata_props:
        if entry.key not in have:
            new = quantized.metadata_props.add()
            new.key, new.value = entry.key, entry.value
    onnx.save(quantized, str(int8_path))
    for temp in (prepared, upgraded):
        if temp.is_file():
            temp.unlink()
    if not onnx_loads(int8_path, input_name, frames[0], imgsz):
        int8_path.unlink()
        raise RuntimeError("quantized model does not load in ONNX Runtime")
    print(f"static INT8 ({'per-channel' if per_channel else 'per-tensor'}, {len(frames)} "
          f"calibration frames) -> {int8_path} ({int8_path.stat().st_size} bytes)", flush=True)
    return int8_path


def onnx_loads(path, input_name=None, sample_path=None, imgsz=640):
    """True if ONNX Runtime can load the model (and run one frame when given)."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if sample_path is not None:
            name = input_name or session.get_inputs()[0].name
            session.run(None, {name: letterbox_tensor(sample_path, imgsz)})
        return True
    except Exception as error:
        print(f"  ONNX Runtime check failed for {Path(path).name}: {str(error)[:300]}", flush=True)
        return False


def export_fasterrcnn_state(path):
    """Save the torchvision COCO weights so pods load them from the registry, not the internet."""
    import torch
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights.get_state_dict(progress=True), str(path))
    print(f"Faster R-CNN state_dict -> {path} ({path.stat().st_size} bytes)", flush=True)
    return path


def prepare_artifacts(pt_path, skip_export=False):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    copied = ARTIFACTS / "yolo26n.pt"
    if Path(pt_path).resolve() != copied.resolve():
        shutil.copy2(pt_path, copied)
    onnx_path = ARTIFACTS / "yolo26n.onnx"
    int8_path = ARTIFACTS / "yolo26n.int8-static.onnx"
    frcnn_path = ARTIFACTS / "fasterrcnn_mv3_320.pth"
    if not skip_export:
        if not onnx_path.is_file():
            export_yolo_onnx(copied, onnx_path, imgsz=640)
        if int8_path.is_file() and not onnx_loads(int8_path):
            print(f"{int8_path.name} does not load; re-exporting", flush=True)
            int8_path.unlink()
        if onnx_path.is_file() and not int8_path.is_file():
            try:
                export_yolo_int8(onnx_path, int8_path)
            except Exception as error:
                print(f"INT8 quantization failed ({error}); skipping that variant", flush=True)
        if not frcnn_path.is_file():
            try:
                export_fasterrcnn_state(frcnn_path)
            except Exception as error:
                print(f"Faster R-CNN export failed ({error}); skipping that variant", flush=True)
    available = {"pytorch": copied}
    if onnx_path.is_file():
        available["onnx"] = onnx_path
    if int8_path.is_file():
        available["onnx-int8"] = int8_path
    if frcnn_path.is_file():
        available["torchvision"] = frcnn_path
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
            "detector": variant["detector"],
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
