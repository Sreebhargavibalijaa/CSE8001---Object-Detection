"""Detection gRPC server. Variant is chosen with MODEL_NAME (no image rebuild)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from services import mot_pb2, mot_pb2_grpc  # noqa: E402
from services.lib.batcher import RequestBatcher  # noqa: E402
from services.lib.metrics import StageMetrics  # noqa: E402
from services.lib.server import serve  # noqa: E402
from services.lib.variant import load_variant  # noqa: E402


def decode_frame(frame):
    image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"bad jpeg stream={frame.stream_id} frame={frame.frame_id}")
    return image


def box_ok(box, width, height):
    if not (0.0 <= box.confidence <= 1.0):
        return False
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return False
    if width and height:
        if box.x1 < -1 or box.y1 < -1 or box.x2 > width + 1 or box.y2 > height + 1:
            return False
    return True


class DetectorService(mot_pb2_grpc.DetectorServicer):
    def __init__(self, predict, metrics, batcher):
        self.predict = predict
        self.metrics = metrics
        self.batcher = batcher

    def Detect(self, request, context):
        started = time.perf_counter()
        try:
            response = self.batcher.submit(request)
            self.metrics.observe(time.perf_counter() - started, True)
            return response
        except Exception as error:
            self.metrics.observe(time.perf_counter() - started, False)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(error))
            raise

    def handle_batch(self, requests):
        frames = [frame for request in requests for frame in request.frames]
        images = [decode_frame(frame) for frame in frames]
        boxes_per_image = self.predict(images) if images else []
        grouped = []
        offset = 0
        for request in requests:
            detections = []
            for frame in request.frames:
                boxes = boxes_per_image[offset]
                offset += 1
                passed = failed = 0
                proto_boxes = []
                for x1, y1, x2, y2, conf in boxes:
                    box = mot_pb2.Box(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                    if box_ok(box, frame.width, frame.height):
                        proto_boxes.append(box)
                        passed += 1
                    else:
                        failed += 1
                self.metrics.assertion(passed, failed)
                detections.append(mot_pb2.FrameDetections(
                    stream_id=frame.stream_id, frame_id=frame.frame_id, boxes=proto_boxes))
            grouped.append(mot_pb2.DetectResponse(detections=detections))
        return grouped


def build_predict(weights, imgsz, conf, device):
    from ultralytics import YOLO
    if "int8" in Path(weights).stem.lower():
        device = "cpu"
    model = YOLO(str(weights), task="detect")

    def predict(images):
        results = model.predict(source=images, classes=[0], device=device, imgsz=imgsz,
                                conf=conf, verbose=False)
        out = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            out.append([(*map(float, xyxy), float(score))
                        for xyxy, score in zip(boxes.xyxy, boxes.conf)])
        return out
    return predict


def main():
    weights, meta = load_variant("detection")
    imgsz = int(os.environ.get("IMGSZ", meta.get("imgsz") or 640))
    conf = float(os.environ.get("CONF", 0.01))
    device = os.environ.get("DEVICE", "cpu")
    metrics = StageMetrics("detect")
    service = DetectorService(build_predict(weights, imgsz, conf, device), metrics, None)
    service.batcher = RequestBatcher(
        service.handle_batch,
        max_batch=int(os.environ.get("BATCH_SIZE", 4)),
        timeout=float(os.environ.get("BATCH_TIMEOUT", 0.05)),
        metrics=metrics,
    )
    print(f"detect MODEL_NAME={os.environ.get('MODEL_NAME')} imgsz={imgsz} "
          f"batch={os.environ.get('BATCH_SIZE', 4)}", flush=True)
    serve(lambda server: mot_pb2_grpc.add_DetectorServicer_to_server(service, server))


if __name__ == "__main__":
    main()
