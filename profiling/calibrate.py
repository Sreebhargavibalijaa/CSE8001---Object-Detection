"""T5: set the SLA from the Guide v4 calibration procedure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from loadgen.open_loop import run_open_loop, summarize, max_rate_meeting_slo  # noqa: E402
from loadgen.pool import stream_frames  # noqa: E402
from profiling.runtime import make_detect_pool, make_track_pool  # noqa: E402

REFERENCE = {
    "detect": "yolo26n_imgsz640",
    "track": "bytetrack",
    "batch": 1,
    "replicas": 1,
    "cores": 1,
}


def e2e_sender(detect_pool, track_pool):
    def send(req):
        det = detect_pool.submit(req, req.stream_id)
        trk = track_pool.submit(
            {"stream_id": req.stream_id, "frame_id": req.frame_id, "boxes": det["boxes"]},
            req.stream_id,
        )
        return {
            "queue_s": (det.get("queue_s") or 0) + (trk.get("queue_s") or 0),
            "infer_s": (det.get("infer_s") or 0) + (trk.get("infer_s") or 0),
            "serdes_s": (det.get("serdes_s") or 0) + (trk.get("serdes_s") or 0),
            "result": trk.get("tracks"),
        }
    return send


def load_pool(frames, replays):
    """Labeled test-slice frames, cycled with new stream_ids. Do not slice to
    `frames` — that would drop sequences and make high-rate bursts too short
    to queue."""
    return stream_frames(max_frames=frames, replays=replays)


def calibrate(frames=8, low_rps=1.0, out=None, replays=2):
    items = load_pool(frames, replays)
    print(f"reference {REFERENCE}  n={len(items)}  low load {low_rps} rps", flush=True)
    detect_pool = make_detect_pool(REFERENCE["detect"], REFERENCE["batch"],
                                   REFERENCE["replicas"], REFERENCE["cores"])
    track_pool = make_track_pool(REFERENCE["track"], REFERENCE["batch"],
                                 REFERENCE["replicas"], REFERENCE["cores"])
    send = e2e_sender(detect_pool, track_pool)
    print("==> warmup (exclude cold-start from L0)", flush=True)
    run_open_loop(send, items[: min(4, len(items))], max(low_rps, 1.0), workers=32)
    track_pool = make_track_pool(REFERENCE["track"], REFERENCE["batch"],
                                 REFERENCE["replicas"], REFERENCE["cores"])
    send = e2e_sender(detect_pool, track_pool)
    print("==> L0 at very low load", flush=True)
    low = summarize(run_open_loop(send, items, low_rps, workers=32))
    if low["errors"]:
        raise RuntimeError(f"low-load probe failed ({low['errors']} errors): {low}")
    l0 = low["p95_s"]
    if l0 is None:
        raise RuntimeError(f"low-load p95 is empty: {low}")
    slo = 2.0 * l0
    print(f"L0 p95={l0*1000:.1f} ms  SLO={slo*1000:.1f} ms", flush=True)
    print("==> T0 = max rate with p95 ≤ SLO", flush=True)
    # Recreate tracker so stream state is clean.
    track_pool = make_track_pool(REFERENCE["track"], REFERENCE["batch"],
                                 REFERENCE["replicas"], REFERENCE["cores"])
    send = e2e_sender(detect_pool, track_pool)
    t0, t0_summary = max_rate_meeting_slo(send, items, slo, low=low_rps, high=None, rounds=5, workers=64)
    target = 4.0 * t0 if t0 > 0 else 4.0 * low_rps
    sla = {
        "guide": "Pipeline Evaluation Guide v4 calibration",
        "hardware": (
            "this machine's CPUs with explicit torch/OMP threads = cores/replica; "
            "YOLO device=cpu (not MPS). Apple Accelerate may still speed kernels; "
            "k8s/Linux numbers are slower (~300 ms detect). Not a GPU SLA."
        ),
        "reference": REFERENCE,
        "request": "one frame of one MOT17 test-slice stream",
        "unit": "milliseconds per frame (p95, end to end) and frames/s",
        "native_fps": 30,
        "n_frames": len(items),
        "L0_p95_ms": l0 * 1000,
        "slo_p95_ms": slo * 1000,
        "slo_p95_s": slo,
        "T0_rps": t0,
        "target_rps": target,
        "target_note": (
            "Target is 4×T0 so a single 1-core replica cannot meet it. "
            "CPU YOLO cannot sustain native 30 FPS; the SLA is in frames/s, not 30 FPS real time."
        ),
        "low_load": low,
        "T0_summary": t0_summary,
        "load_pool": "eval_slices/MOT17_test  sequences 02/04/09-FRCNN second halves",
    }
    out = Path(out or ROOT / "docs" / "sla.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sla, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: sla[k] for k in ("L0_p95_ms", "slo_p95_ms", "T0_rps", "target_rps")}, indent=2))
    print(f"wrote {out}")
    return sla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--low-rps", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "sla.json")
    args = parser.parse_args()
    calibrate(args.frames, args.low_rps, args.out, args.replays)


if __name__ == "__main__":
    main()
