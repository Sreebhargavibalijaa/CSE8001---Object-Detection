"""Send a MOT17 sequence through detect -> track -> evaluate over gRPC.

Tracker address is chosen from stream_id so frames of one video stay on one
StatefulSet pod, in order. Detection is stateless.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import grpc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from common import image_frames, load_sequence  # noqa: E402
from services import mot_pb2, mot_pb2_grpc  # noqa: E402
from services.lib.routing import tracker_target  # noqa: E402


def stub(channel_cls, address):
    channel = grpc.insecure_channel(address, options=[("grpc.max_send_message_length", 32 * 1024 * 1024),
                                                      ("grpc.max_receive_message_length", 32 * 1024 * 1024)])
    return channel, channel_cls(channel)


def encode(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"jpeg encode failed: {path}")
    h, w = image.shape[:2]
    return bytes(buffer), w, h


def main():
    parser = argparse.ArgumentParser(description="gRPC MOT17 pipeline client")
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", str(ROOT / "MOT17")))
    parser.add_argument("--sequence", default=os.environ.get("SEQUENCE", "MOT17-02-DPM"))
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("MAX_FRAMES", 30)))
    parser.add_argument("--detect-addr", default=os.environ.get("DETECT_ADDR", "127.0.0.1:50051"))
    parser.add_argument("--evaluate-addr", default=os.environ.get("EVALUATE_ADDR", "127.0.0.1:50053"))
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    frames = image_frames(sequence, args.max_frames)
    detect_ch, detector = stub(mot_pb2_grpc.DetectorStub, args.detect_addr)
    eval_ch, evaluator = stub(mot_pb2_grpc.EvaluatorStub, args.evaluate_addr)
    track_stubs = {}
    all_tracks = []
    started = time.perf_counter()
    for frame_id, path in frames:
        jpeg, width, height = encode(path)
        detected = detector.Detect(mot_pb2.DetectRequest(frames=[
            mot_pb2.Frame(stream_id=sequence.name, frame_id=frame_id, jpeg=jpeg,
                          width=width, height=height)]))
        boxes = detected.detections[0].boxes if detected.detections else []
        target, shard, replicas = tracker_target(sequence.name)
        if target not in track_stubs:
            channel, track_stubs[target] = stub(mot_pb2_grpc.TrackerStub, target)
            print(f"stream {sequence.name} -> tracker {target} (shard {shard}/{replicas})", flush=True)
        tracked = track_stubs[target].Track(mot_pb2.TrackRequest(
            stream_id=sequence.name, frame_id=frame_id, width=width, height=height, boxes=boxes))
        all_tracks.extend(tracked.tracks)
        if frame_id % 10 == 0 or frame_id == len(frames):
            print(f"frame {frame_id}/{len(frames)}  dets={len(boxes)}  tracks={len(tracked.tracks)}",
                  flush=True)
    scored = evaluator.Evaluate(mot_pb2.EvaluateRequest(
        stream_id=sequence.name, width=sequence.width, height=sequence.height, tracks=all_tracks))
    elapsed = time.perf_counter() - started
    print(f"done {len(frames)} frames in {elapsed:.1f}s ({len(frames) / elapsed:.2f} FPS)")
    print(f"motion pass {scored.assertion_pass_rate:.1f}%  "
          f"({scored.assertions_failed}/{scored.assertions_checked} failed)  "
          f"HOTA {scored.hota:.1f}  MOTA {scored.mota:.1f}  Recall {scored.recall:.1f}")
    detect_ch.close()
    eval_ch.close()


if __name__ == "__main__":
    main()
