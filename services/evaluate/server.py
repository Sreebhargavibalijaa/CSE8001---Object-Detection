"""Evaluation gRPC server: motion assertion always, TrackEval when GT is mounted."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import grpc

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
sys.path[:0] = [str(ROOT), str(PIPELINE)]

from common import load_sequence, write_tracks  # noqa: E402
from evaluate import MOTION_THRESHOLD, motion_assertion, tracking_metrics  # noqa: E402
from services import mot_pb2, mot_pb2_grpc  # noqa: E402
from services.lib.batcher import RequestBatcher  # noqa: E402
from services.lib.metrics import StageMetrics  # noqa: E402
from services.lib.server import serve  # noqa: E402


class SequenceShim:
    """Minimal stand-in for common.Sequence used by the assertion code."""

    def __init__(self, name, width, height, seq_length=1):
        self.name = name
        self.width = width
        self.height = height
        self.seq_length = max(1, int(seq_length))

    def frame_limit(self, max_frames=None):
        if max_frames is None:
            return self.seq_length
        return min(int(max_frames), self.seq_length)


class EvaluatorService(mot_pb2_grpc.EvaluatorServicer):
    def __init__(self, metrics, batcher, dataset_root):
        self.metrics = metrics
        self.batcher = batcher
        self.dataset_root = dataset_root

    def Evaluate(self, request, context):
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
        return [self._evaluate_one(request) for request in requests]

    def _evaluate_one(self, request):
        rows = [(t.frame_id, t.track_id, t.left, t.top, t.width, t.height, t.confidence)
                for t in request.tracks]
        shim = SequenceShim(request.stream_id, request.width or 1, request.height or 1,
                            seq_length=max((t.frame_id for t in request.tracks), default=1))
        motion = motion_assertion(shim, rows, MOTION_THRESHOLD)
        passed = (motion["assertions_checked"] or 0) - (motion["assertions_failed"] or 0)
        self.metrics.assertion(passed, motion["assertions_failed"] or 0)
        hota = mota = idf1 = recall = precision = 0.0
        if self.dataset_root:
            try:
                sequence = load_sequence(self.dataset_root, request.stream_id)
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
                    path = Path(handle.name)
                write_tracks(path, rows)
                scored = tracking_metrics(sequence, path, max_frames=max((t.frame_id for t in request.tracks), default=1))
                path.unlink(missing_ok=True)
                hota, mota, idf1 = scored["HOTA"], scored["MOTA"], scored["IDF1"]
                recall, precision = scored["Recall"], scored["Precision"]
            except Exception as error:
                print(f"TrackEval skipped: {error}", flush=True)
        return mot_pb2.EvaluateResponse(
            assertions_checked=int(motion["assertions_checked"] or 0),
            assertions_failed=int(motion["assertions_failed"] or 0),
            assertion_pass_rate=float(motion["assertion_pass_rate"] or 0.0),
            hota=hota, mota=mota, idf1=idf1, recall=recall, precision=precision,
        )


def main():
    metrics = StageMetrics("evaluate")
    service = EvaluatorService(
        metrics, None, os.environ.get("DATASET_ROOT", str(ROOT / "MOT17")))
    service.batcher = RequestBatcher(
        service.handle_batch,
        max_batch=int(os.environ.get("BATCH_SIZE", 1)),
        timeout=float(os.environ.get("BATCH_TIMEOUT", 0.05)),
        metrics=metrics,
    )
    print("evaluate service ready", flush=True)
    serve(lambda server: mot_pb2_grpc.add_EvaluatorServicer_to_server(service, server))


if __name__ == "__main__":
    main()
