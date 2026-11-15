"""Step 1: run one pretrained person detector over a MOT17 sequence and save the boxes.

Writes <output-root>/detections/<sequence>/<detector-tag>.csv

Faster R-CNN is the MobileNetV3-Large 320-FPN variant. The ResNet-50 FPN model
was dropped: at ~0.13 FPS on CPU it makes a 7-sequence profiling sweep impractical.
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from common import REPO_ROOT, add_common_arguments, image_frames, load_sequence, resolve_device, write_detections

DETR_MODEL = "facebook/detr-resnet-50"
FASTER_RCNN_TAG = "fasterrcnn_mv3_320"


def detector_tag(detector, imgsz=640, frame_skip=1, runtime="pytorch"):
    if detector == "yolo26n":
        tag = f"yolo26n_imgsz{imgsz}" if runtime == "pytorch" else f"yolo26n_{runtime}_imgsz{imgsz}"
        if frame_skip > 1:
            tag += f"_skip{frame_skip}"
        return tag
    if detector == "fasterrcnn":
        return FASTER_RCNN_TAG
    return "detr"


def yolo_detector(weights, conf, imgsz, device):
    from ultralytics import YOLO
    # Dynamic INT8 graphs use integer ops that CoreML/MPS EPs do not implement.
    if "int8" in Path(weights).stem.lower():
        device = "cpu"
    model = YOLO(str(weights), task="detect")

    def detect(path):
        boxes = model.predict(source=str(path), classes=[0], device=device, imgsz=imgsz,
                              conf=conf, verbose=False)[0].boxes.cpu().numpy()
        return [(*map(float, box), float(score)) for box, score in zip(boxes.xyxy, boxes.conf)]
    return detect


def fasterrcnn_detector(conf, device):
    """MobileNetV3-Large 320-FPN Faster R-CNN — same two-stage family, CPU-practical."""
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None,
        num_classes=len(weights.meta["categories"]), box_score_thresh=conf)
    model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    model.to(device).eval()
    preprocess = weights.transforms()
    person = weights.meta["categories"].index("person")
    torch_device = torch.device(device)

    def detect(path):
        with Image.open(path) as image:
            tensor = preprocess(image.convert("RGB")).to(torch_device)
        prediction = {key: value.detach().cpu() for key, value in model([tensor])[0].items()}
        return person_boxes(prediction, person, conf)
    return detect


def detr_detector(model_name, conf, device):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(device).eval()
    person = next(int(key) for key, value in model.config.id2label.items() if value.lower() == "person")

    def detect(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        prediction = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[image.height, image.width]]), threshold=0.0)[0]
        prediction = {key: value.detach().cpu() for key, value in prediction.items()}
        return person_boxes(prediction, person, conf)
    return detect


def person_boxes(prediction, person_label, conf):
    return [
        (*map(float, box), float(score))
        for box, score, label in zip(prediction["boxes"].tolist(), prediction["scores"].tolist(),
                                     prediction["labels"].tolist())
        if label == person_label and score >= conf
    ]


def build_detector(detector, conf, device, imgsz=640, yolo_weights=None, detr_model=DETR_MODEL,
                   frame_skip=1, runtime=None, tag=None):
    device = resolve_device(device)
    weights = Path(yolo_weights) if yolo_weights else REPO_ROOT / "yolo26n.pt"
    if runtime is None:
        runtime = "onnx-int8" if "int8" in weights.stem.lower() else (
            "onnx" if weights.suffix == ".onnx" else "pytorch")
    if detector == "yolo26n":
        return (tag or detector_tag(detector, imgsz, frame_skip, runtime),
                yolo_detector(weights, conf, imgsz, device))
    if detector == "fasterrcnn":
        # torchvision Faster R-CNN is far slower on Apple MPS than on CPU.
        if device == "mps":
            print("fasterrcnn: using cpu (MPS is slower for this model)", flush=True)
            device = "cpu"
        return FASTER_RCNN_TAG, fasterrcnn_detector(conf, device)
    if detector == "detr":
        return "detr", detr_detector(detr_model, conf, device)
    raise ValueError(f"Unknown detector: {detector}")


def run_detection(sequence, detect, output, max_frames=None, label="detector", frame_skip=1):
    frames = image_frames(sequence, max_frames)
    rows = []
    started = perf_counter()
    with torch.inference_mode():
        for frame, path in frames:
            if frame_skip > 1 and (frame - 1) % frame_skip != 0:
                continue
            rows.extend((frame, *box) for box in detect(path))
            if frame % 25 == 0 or frame == len(frames):
                print(f"{label}: {sequence.name} frame {frame}/{len(frames)}", flush=True)
    elapsed = perf_counter() - started
    write_detections(output, rows)
    fps = len(frames) / elapsed if elapsed else 0.0
    print(f"{len(rows)} person detections over {len(frames)} frames "
          f"({fps:.2f} FPS) -> {output}")
    return fps, len(rows), len(frames)


def main():
    parser = argparse.ArgumentParser(description="Person detection only; no tracker is run")
    add_common_arguments(parser)
    parser.add_argument("--detector", choices=["yolo26n", "fasterrcnn", "detr"], default="yolo26n")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Keep every box above this score; the tracker applies its own thresholds")
    parser.add_argument("--yolo-weights", type=Path, default=REPO_ROOT / "yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size; 640 misses many MOT17 pedestrians")
    parser.add_argument("--frame-skip", type=int, default=1, help="Run the detector every Nth frame")
    parser.add_argument("--tag", help="Override the detections CSV stem")
    parser.add_argument("--detr-model", default=DETR_MODEL, help="Hugging Face model id or local directory")
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    device = resolve_device(args.device)
    tag, detect = build_detector(args.detector, args.conf, device, args.imgsz,
                                 args.yolo_weights, args.detr_model, args.frame_skip, tag=args.tag)
    output = args.output_root / "detections" / sequence.name / f"{tag}.csv"
    print(f"detector={tag} device={device} frame_skip={args.frame_skip}", flush=True)
    run_detection(sequence, detect, output, args.max_frames, tag, args.frame_skip)


if __name__ == "__main__":
    main()
