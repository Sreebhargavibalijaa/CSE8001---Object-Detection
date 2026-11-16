"""Tracking gRPC server. One in-memory tracker per stream_id, frames in order."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from services import mot_pb2, mot_pb2_grpc  # noqa: E402
from services.lib.batcher import RequestBatcher  # noqa: E402
from services.lib.metrics import StageMetrics  # noqa: E402
from services.lib.server import serve, timing_metadata  # noqa: E402
from services.lib.variant import load_variant  # noqa: E402
from track import FrameDetections, build_tracker  # noqa: E402


class StreamState:
    def __init__(self, tracker_name, config_path):
        self.tracker = build_tracker(tracker_name, config_path)
        self.last = {}  # track_id -> (frame, cx, cy, height)
        self.last_count = None
        self.lock = threading.Lock()


class TrackerService(mot_pb2_grpc.TrackerServicer):
    def __init__(self, tracker_name, config_path, metrics, batcher):
        self.tracker_name = tracker_name
        self.config_path = config_path
        self.metrics = metrics
        self.batcher = batcher
        self.streams = {}
        self.streams_lock = threading.Lock()

    def _state(self, stream_id):
        with self.streams_lock:
            if stream_id not in self.streams:
                self.streams[stream_id] = StreamState(self.tracker_name, self.config_path)
            return self.streams[stream_id]

    def Track(self, request, context):
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
        # Cross-stream batching is allowed; within a stream, keep frame order.
        ordered = sorted(enumerate(requests), key=lambda item: (item[1].stream_id, item[1].frame_id))
        placed = [None] * len(requests)
        for index, request in ordered:
            placed[index] = self._track_one(request)
        return placed

    def _track_one(self, request):
        state = self._state(request.stream_id)
        values = [[box.x1, box.y1, box.x2, box.y2, box.confidence, 0.0] for box in request.boxes]
        with state.lock:
            outputs = state.tracker.update(FrameDetections(values))
            tracks = []
            current = {}
            a1 = a2 = a3 = True
            for x1, y1, x2, y2, track_id, confidence, *_ in outputs:
                track_id = int(track_id)
                width, height = float(x2 - x1), float(y2 - y1)
                cx, cy = (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
                if width <= 0 or not (1.0 <= height / width <= 6.0):
                    a2 = False
                prev = state.last.get(track_id)
                if prev is not None and request.frame_id > prev[0]:
                    if math.hypot(cx - prev[1], cy - prev[2]) > 0.5 * height:
                        a1 = False
                current[track_id] = (request.frame_id, cx, cy, height)
                tracks.append(mot_pb2.Track(
                    frame_id=request.frame_id, track_id=track_id,
                    left=float(x1), top=float(y1), width=width, height=height,
                    confidence=float(confidence)))
            count = len(current)
            if state.last_count is not None and abs(count - state.last_count) > max(3, 0.3 * state.last_count):
                a3 = False
            state.last = current
            state.last_count = count
            passed, failed = (1, 0) if (a1 and a2 and a3) else (0, 1)
        self.metrics.assertion(passed, failed)
        return mot_pb2.TrackResponse(tracks=tracks, assertions_passed=passed, assertions_failed=failed)


def main():
    config_path, meta = load_variant("tracking")
    tracker_name = meta.get("tracker") or os.environ.get("TRACKER_TYPE", "bytetrack")
    metrics = StageMetrics("track")
    service = TrackerService(tracker_name, config_path, metrics, None)
    service.batcher = RequestBatcher(
        service.handle_batch,
        max_batch=int(os.environ.get("BATCH_SIZE", 8)),
        timeout=float(os.environ.get("BATCH_TIMEOUT", 0.02)),
        metrics=metrics,
    )
    print(f"track MODEL_NAME={os.environ.get('MODEL_NAME')} type={tracker_name} "
          f"batch={os.environ.get('BATCH_SIZE', 8)}", flush=True)
    serve(lambda server: mot_pb2_grpc.add_TrackerServicer_to_server(service, server))


if __name__ == "__main__":
    main()
