"""Step 1: run one pretrained person detector over a MOT17 sequence and save the boxes.

Writes <output-root>/detections/<sequence>/<detector>.csv
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from common import REPO_ROOT, add_common_arguments, image_frames, load_sequence, write_detections

DETR_MODEL = "facebook/detr-resnet-50"


def yolo_detector(weights, conf, imgsz):
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")

    def detect(path):
        boxes = model.predict(source=str(path), classes=[0], device="cpu", imgsz=imgsz,
                              conf=conf, verbose=False)[0].boxes.cpu().numpy()
        return [(*map(float, box), float(score)) for box, score in zip(boxes.xyxy, boxes.conf)]
    return detect


def fasterrcnn_detector(conf):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # Built without weights and loaded strictly afterwards, as in our reported runs.
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    num_classes=len(weights.meta["categories"]), box_score_thresh=conf)
    model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    model.eval()
    preprocess = weights.transforms()
    person = weights.meta["categories"].index("person")

    def detect(path):
        with Image.open(path) as image:
            prediction = model([preprocess(image.convert("RGB"))])[0]
        return person_boxes(prediction, person, conf)
    return detect


def detr_detector(model_name, conf):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name).eval()
    person = next(int(key) for key, value in model.config.id2label.items() if value.lower() == "person")

    def detect(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
        outputs = model(**processor(images=image, return_tensors="pt"))
        # target_sizes rescales the boxes back to original-image pixels.
        prediction = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[image.height, image.width]]), threshold=0.0)[0]
        return person_boxes(prediction, person, conf)
    return detect


def person_boxes(prediction, person_label, conf):
    return [
        (*map(float, box), float(score))
        for box, score, label in zip(prediction["boxes"].tolist(), prediction["scores"].tolist(),
                                     prediction["labels"].tolist())
        if label == person_label and score >= conf
    ]


def main():
    parser = argparse.ArgumentParser(description="Person detection only; no tracker is run")
    add_common_arguments(parser)
    parser.add_argument("--detector", choices=["yolo26n", "fasterrcnn", "detr"], default="yolo26n")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Keep every box above this score; the tracker applies its own thresholds")
    parser.add_argument("--yolo-weights", type=Path, default=REPO_ROOT / "yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO only")
    parser.add_argument("--detr-model", default=DETR_MODEL, help="Hugging Face model id or local directory")
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    frames = image_frames(sequence, args.max_frames)
    if args.detector == "yolo26n":
        detect = yolo_detector(args.yolo_weights, args.conf, args.imgsz)
    elif args.detector == "fasterrcnn":
        detect = fasterrcnn_detector(args.conf)
    else:
        detect = detr_detector(args.detr_model, args.conf)

    rows = []
    started = perf_counter()
    with torch.inference_mode():
        for frame, path in frames:
            rows.extend((frame, *box) for box in detect(path))
            if frame % 25 == 0 or frame == len(frames):
                print(f"{args.detector}: frame {frame}/{len(frames)}", flush=True)
    elapsed = perf_counter() - started

    output = args.output_root / "detections" / sequence.name / f"{args.detector}.csv"
    write_detections(output, rows)
    print(f"{len(rows)} person detections over {len(frames)} frames "
          f"({len(frames) / elapsed:.2f} FPS) -> {output}")


if __name__ == "__main__":
    main()
