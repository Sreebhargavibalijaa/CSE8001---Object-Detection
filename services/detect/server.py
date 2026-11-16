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
from services.lib.server import serve, timing_metadata  # noqa: E402
from services.lib.variant import load_variant  # noqa: E402


CPU_THREADS = 0  # set by limit_cpu_threads(); 0 = leave library defaults


def apply_torch_threads():
    """Re-apply the thread cap in the calling thread.

    Ultralytics calls torch.set_num_threads(8) while setting up the first
    predict, and the OpenMP thread count is per calling thread, so the cap has
    to be set again in the batcher thread before every model call. Measured on
    the VM: 8 threads in a 1-core pod were throttled in 74% of CPU periods
    (about 850 ms per frame instead of about 100 ms).
    """
    if CPU_THREADS > 0:
        import torch
        if torch.get_num_threads() != CPU_THREADS:
            torch.set_num_threads(CPU_THREADS)


def cgroup_cpu_limit():
    """CPU limit of this container in whole cores (0 if unlimited or unknown)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota == "max":
            return 0
        return max(1, int(float(quota) / float(period) + 0.999))
    except (OSError, ValueError):
        return 0


def limit_cpu_threads():
    """Match torch / OpenCV / ONNX Runtime threads to the pod CPU limit.

    Without this each library starts one thread per *host* core (16 on the VM),
    which thrashes under a 1- or 2-core Kubernetes CPU limit.
    """
    global CPU_THREADS
    n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if n <= 0:
        n = cgroup_cpu_limit()
    if n <= 0:
        return None
    CPU_THREADS = n
    import torch
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    try:
        import onnxruntime as ort
        base = ort.InferenceSession

        class LimitedSession(base):
            def __init__(self, path_or_bytes, sess_options=None, providers=None,
                         provider_options=None, **kwargs):
                options = sess_options or ort.SessionOptions()
                options.intra_op_num_threads = n
                options.inter_op_num_threads = 1
                super().__init__(path_or_bytes, sess_options=options, providers=providers,
                                 provider_options=provider_options, **kwargs)

        ort.InferenceSession = LimitedSession
    except ImportError:
        pass
    print(f"cpu threads limited to {n}", flush=True)
    return n


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
            response, queue_s, infer_s = self.batcher.submit_timed(request)
            self.metrics.observe(time.perf_counter() - started, True)
            timing_metadata(context, queue_s, infer_s)
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


def build_predict(weights, imgsz, conf, device, per_image=False):
    from ultralytics import YOLO
    if "int8" in Path(weights).stem.lower():
        device = "cpu"
    model = YOLO(str(weights), task="detect")

    def run(source):
        apply_torch_threads()
        results = model.predict(source=source, classes=[0], device=device, imgsz=imgsz,
                                conf=conf, verbose=False)
        out = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            out.append([(*map(float, xyxy), float(score))
                        for xyxy, score in zip(boxes.xyxy, boxes.conf)])
        return out

    def predict(images):
        if per_image:
            # ONNX exports have a fixed batch of 1: run the batch image by image.
            return [run(image)[0] for image in images]
        return run(images)
    return predict


def build_fasterrcnn_predict(weights, conf):
    """torchvision Faster R-CNN MobileNetV3-Large 320-FPN from a registry state_dict."""
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.transforms.functional import to_tensor
    categories = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta["categories"]
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=len(categories), box_score_thresh=conf)
    model.load_state_dict(torch.load(str(weights), map_location="cpu"), strict=True)
    model.eval()
    person = categories.index("person")

    def predict(images):
        apply_torch_threads()
        tensors = [to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        with torch.inference_mode():
            outputs = model(tensors)
        out = []
        for output in outputs:
            out.append([(*map(float, box), float(score))
                        for box, score, label in zip(output["boxes"].tolist(),
                                                     output["scores"].tolist(),
                                                     output["labels"].tolist())
                        if label == person and score >= conf])
        return out
    return predict


def main():
    limit_cpu_threads()
    weights, meta = load_variant("detection")
    imgsz = int(os.environ.get("IMGSZ", meta.get("imgsz") or 640))
    conf = float(os.environ.get("CONF", 0.01))
    device = os.environ.get("DEVICE", "cpu")
    runtime = meta.get("runtime", "pytorch")
    if runtime == "torchvision":
        predict = build_fasterrcnn_predict(weights, conf)
    else:
        predict = build_predict(weights, imgsz, conf, device, per_image=runtime.startswith("onnx"))
    # One warm-up frame: loads the model and lets Ultralytics finish its setup
    # (which resets the thread count) before the pod reports Ready.
    predict([np.zeros((1080, 1920, 3), dtype=np.uint8)])
    apply_torch_threads()
    metrics = StageMetrics("detect")
    service = DetectorService(predict, metrics, None)
    service.batcher = RequestBatcher(
        service.handle_batch,
        max_batch=int(os.environ.get("BATCH_SIZE", 4)),
        timeout=float(os.environ.get("BATCH_TIMEOUT", 0.05)),
        metrics=metrics,
    )
    print(f"detect MODEL_NAME={os.environ.get('MODEL_NAME')} runtime={runtime} imgsz={imgsz} "
          f"batch={os.environ.get('BATCH_SIZE', 4)}", flush=True)
    serve(lambda server: mot_pb2_grpc.add_DetectorServicer_to_server(service, server))


if __name__ == "__main__":
    main()
