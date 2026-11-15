"""In-process stage workers with explicit CPU threads and replica count.

Cores/replica = torch/OMP thread cap. Replicas = independent models + batchers
(detect round-robins; tracker hashes stream_id so one stream stays in order).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
sys.path[:0] = [str(PIPELINE), str(ROOT)]

from services.lib.batcher import RequestBatcher  # noqa: E402
from track import FrameDetections, build_tracker  # noqa: E402

ARTIFACTS = PIPELINE / "registry" / "artifacts"


def pin_threads(cores):
    n = max(1, int(round(float(cores))))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(2, n)))
    except RuntimeError:
        pass
    return n


DETECT_VARIANTS = {
    "yolo26n_imgsz640": {
        "detector": "yolo26n", "imgsz": 640, "weights": ROOT / "yolo26n.pt",
        "hota_key": "yolo26n_imgsz640",
    },
    "yolo26n_imgsz1280": {
        "detector": "yolo26n", "imgsz": 1280, "weights": ROOT / "yolo26n.pt",
        "hota_key": "yolo26n_imgsz1280",
    },
    "yolo26n-onnx-imgsz640": {
        "detector": "yolo26n", "imgsz": 640,
        "weights": ARTIFACTS / "detection" / "yolo26n-onnx-imgsz640" / "1.0.0" / "yolo26n.onnx",
        "hota_key": "yolo26n-onnx-imgsz640",
    },
    "yolo26n-onnx-int8-imgsz640": {
        "detector": "yolo26n", "imgsz": 640,
        "weights": ARTIFACTS / "detection" / "yolo26n-onnx-int8-imgsz640" / "1.0.0" / "yolo26n.int8.onnx",
        "hota_key": "yolo26n-onnx-int8-imgsz640",
    },
}

TRACK_VARIANTS = {
    "bytetrack": {"name": "bytetrack", "config": PIPELINE / "configs" / "bytetrack.yaml",
                  "hota_key": "bytetrack"},
    "bytetrack_low": {"name": "bytetrack", "config": PIPELINE / "configs" / "bytetrack_low.yaml",
                      "hota_key": "bytetrack_low"},
    "bytetrack_long": {"name": "bytetrack", "config": PIPELINE / "configs" / "bytetrack_long.yaml",
                       "hota_key": "bytetrack"},
    "ocsort": {"name": "ocsort", "config": PIPELINE / "configs" / "ocsort.yaml",
               "hota_key": "ocsort"},
}

RESOURCE_CONFIGS = (
    {"replicas": 1, "cores": 1},
    {"replicas": 1, "cores": 2},
    {"replicas": 2, "cores": 1},
    {"replicas": 2, "cores": 2},
)
BATCH_SIZES = (1, 4, 8)


def load_detect_fn(variant, cores):
    """Return a callable(image_bgr) -> boxes. One YOLO/ONNX session per replica."""
    pin_threads(cores)
    spec = DETECT_VARIANTS[variant]
    weights = Path(spec["weights"])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    imgsz = spec["imgsz"]

    def detect_image(image_bgr):
        result = model.predict(
            source=image_bgr, classes=[0], device="cpu", imgsz=imgsz,
            conf=0.01, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        return [(*map(float, box), float(score), 0.0)
                for box, score in zip(boxes.xyxy, boxes.conf)]

    detect_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
    return detect_image


class DetectWorker:
    def __init__(self, variant, batch_size, cores, detect_fn=None):
        pin_threads(cores)
        self.detect_one = detect_fn or load_detect_fn(variant, cores)
        self.batcher = RequestBatcher(self._handle, max_batch=batch_size, timeout=0.008)

    def _handle(self, batch):
        images, serdes = [], []
        for item in batch:
            t0 = time.perf_counter()
            image = cv2.imread(str(item.path))
            serdes.append(time.perf_counter() - t0)
            if image is None:
                raise FileNotFoundError(item.path)
            images.append(image)
        infer_start = time.perf_counter()
        boxes = []
        with torch.inference_mode():
            for image in images:
                boxes.append(self.detect_one(image))
        infer = time.perf_counter() - infer_start
        share = infer / max(1, len(batch))
        return [{"boxes": b, "infer_s": share, "serdes_s": s, "queue_s": None}
                for b, s in zip(boxes, serdes)]

    def submit(self, item):
        enq = time.perf_counter()
        result = self.batcher.submit(item)
        total = time.perf_counter() - enq
        result["queue_s"] = max(0.0, total - (result.get("infer_s") or 0) - (result.get("serdes_s") or 0))
        return result


class TrackWorker:
    def __init__(self, variant, batch_size, cores):
        pin_threads(cores)
        spec = TRACK_VARIANTS[variant]
        self.spec = spec
        self.streams = {}
        self.lock = threading.Lock()
        self.batcher = RequestBatcher(self._handle, max_batch=batch_size, timeout=0.004)

    def _tracker(self, stream_id):
        with self.lock:
            if stream_id not in self.streams:
                self.streams[stream_id] = build_tracker(self.spec["name"], self.spec["config"])
            return self.streams[stream_id]

    def _handle(self, batch):
        ordered = sorted(enumerate(batch), key=lambda pair: (pair[1]["stream_id"], pair[1]["frame_id"]))
        placed = [None] * len(batch)
        infer_start = time.perf_counter()
        for index, item in ordered:
            values = item.get("boxes") or []
            tracker = self._tracker(item["stream_id"])
            boxed = [list(row) if len(row) == 6 else list(row)[:5] + [0.0] for row in values]
            outputs = tracker.update(FrameDetections(boxed))
            tracks = [(item["frame_id"], int(row[4]), float(row[0]), float(row[1]),
                       float(row[2] - row[0]), float(row[3] - row[1]), float(row[5]))
                      for row in outputs]
            placed[index] = tracks
        infer = (time.perf_counter() - infer_start) / max(1, len(batch))
        return [{"tracks": tracks, "infer_s": infer, "serdes_s": 0.0, "queue_s": None}
                for tracks in placed]

    def submit(self, item):
        enq = time.perf_counter()
        result = self.batcher.submit(item)
        total = time.perf_counter() - enq
        result["queue_s"] = max(0.0, total - (result.get("infer_s") or 0))
        return result


class ReplicaPool:
    def __init__(self, factory, replicas, sticky=False):
        self.workers = [factory() for _ in range(max(1, replicas))]
        self.sticky = sticky
        self.rr = 0
        self.lock = threading.Lock()

    @classmethod
    def from_workers(cls, workers, sticky=False):
        pool = cls.__new__(cls)
        pool.workers = list(workers)
        pool.sticky = sticky
        pool.rr = 0
        pool.lock = threading.Lock()
        return pool

    def pick(self, stream_id):
        if len(self.workers) == 1:
            return self.workers[0]
        if self.sticky:
            return self.workers[hash(stream_id) % len(self.workers)]
        with self.lock:
            worker = self.workers[self.rr % len(self.workers)]
            self.rr += 1
            return worker

    def submit(self, item, stream_id):
        return self.pick(stream_id).submit(item)


def make_detect_pool(variant, batch_size, replicas, cores, detect_fns=None):
    if detect_fns is None:
        detect_fns = [load_detect_fn(variant, cores) for _ in range(max(1, replicas))]
    workers = [DetectWorker(variant, batch_size, cores, detect_fn=fn) for fn in detect_fns]
    return ReplicaPool.from_workers(workers, sticky=False)


def make_track_pool(variant, batch_size, replicas, cores):
    return ReplicaPool(lambda: TrackWorker(variant, batch_size, cores), replicas, sticky=True)
