"""T7: end-to-end configs, gold-input ablation, SLA flags, Pareto inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from loadgen.open_loop import run_open_loop, summarize  # noqa: E402  # used in live extra scale-out too
from loadgen.pool import cached_detections, gold_detections, stream_frames  # noqa: E402
from profiling.accuracy_lookup import e2e_hota  # noqa: E402
from profiling.runtime import (  # noqa: E402
    DetectWorker, ReplicaPool, load_detect_fn, make_track_pool, pin_threads,
)

DETECT = ("yolo26n_imgsz640", "yolo26n_imgsz1280", "yolo26n-onnx-imgsz640", "yolo26n-onnx-int8-imgsz640")
TRACK = ("bytetrack", "bytetrack_low", "bytetrack_long", "ocsort")
LIVE_TRACK = ("bytetrack", "ocsort")

# (detect_replicas, detect_cores, track_replicas, track_cores)
TOPOLOGIES = (
    (1, 1, 1, 1),
    (2, 1, 1, 1),
)
E2E_BATCH = 4


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_dict(detect, track, d_r, d_c, t_r, t_c, summary, hota, note, source, target, slo_ms):
    cost = d_r * d_c + t_r * t_c
    return {
        "detect": detect, "track": track, "batch": E2E_BATCH,
        "detect_replicas": d_r, "detect_cores": d_c,
        "track_replicas": t_r, "track_cores": t_c,
        "replicas": d_r, "cores_per_replica": d_c,
        "cost": cost,
        "p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"], "p99_ms": summary["p99_ms"],
        "queue_p95_ms": summary["queue_p95_ms"], "infer_p95_ms": summary["infer_p95_ms"],
        "serdes_p95_ms": summary.get("serdes_p95_ms"),
        "meets_sla": bool(summary["p95_ms"] is not None and summary["p95_ms"] <= slo_ms
                          and summary["errors"] == 0),
        "hota": hota, "accuracy_note": note, "source": source,
        "target_rps": target, "slo_p95_ms": slo_ms, "errors": summary["errors"],
    }


def live_pair(detect_name, track_name, items, rate, detect_fn, d_r, d_c, t_r, t_c):
    pin_threads(d_c)
    fns = [detect_fn] if d_r == 1 else [detect_fn, load_detect_fn(detect_name, d_c)]
    detect_pool = ReplicaPool.from_workers(
        [DetectWorker(detect_name, E2E_BATCH, d_c, detect_fn=fn) for fn in fns], sticky=False)
    track_pool = make_track_pool(track_name, E2E_BATCH, t_r, t_c)

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

    return summarize(run_open_loop(send, items, rate, workers=64))


def cached_pair(track_name, items, rate, boxes_by_seq, t_r, t_c):
    pin_threads(t_c)
    track_pool = make_track_pool(track_name, E2E_BATCH, t_r, t_c)

    def send(req):
        boxes = boxes_by_seq.get(req.sequence, {}).get(req.frame_id, [])
        return track_pool.submit(
            {"stream_id": req.stream_id, "frame_id": req.frame_id, "boxes": boxes},
            req.stream_id,
        )

    return summarize(run_open_loop(send, items, rate, workers=64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sla", type=Path, default=ROOT / "docs" / "sla.json")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "profile")
    args = parser.parse_args()
    sla = json.loads(args.sla.read_text(encoding="utf-8"))
    slo_ms = float(sla["slo_p95_ms"])
    target = float(sla["target_rps"])
    items = stream_frames(max_frames=args.frames, replays=args.replays)
    rows = []

    print("==> gold-input ablation (GT boxes → tracker, no detector)", flush=True)
    gold = gold_detections()
    for track in TRACK:
        print(f"  gold + {track}", flush=True)
        hota, note = e2e_hota("gold_gt", "bytetrack")
        summary = cached_pair(track, items, target, gold, 1, 1)
        rows.append(row_dict("gold_gt", track, 0, 0, 1, 1, summary, hota, note,
                             "gold_input", target, slo_ms))

    print("==> cached-detect configs (accuracy table; tracker latency only)", flush=True)
    for detect in DETECT:
        try:
            boxes = cached_detections(detect)
        except FileNotFoundError:
            boxes = cached_detections("yolo26n_imgsz640")
        for track in TRACK:
            print(f"  cached {detect} + {track}", flush=True)
            try:
                hota, note = e2e_hota(detect, track)
            except KeyError:
                hota, note = e2e_hota("yolo26n_imgsz640", "bytetrack")
                note = f"HOTA lookup fallback; {detect}+{track}"
            summary = cached_pair(track, items, target, boxes, 1, 1)
            rows.append(row_dict(detect, track, 0, 0, 1, 1, summary, hota,
                                 note + " (cached upstream; SLA from live rows)",
                                 "cached_detect", target, slo_ms))

    print("==> live detect+track at SLA target rate (two topologies)", flush=True)
    detect_cache = {}
    for detect in DETECT:
        detect_cache[detect] = load_detect_fn(detect, 1)
        for track in LIVE_TRACK:
            for d_r, d_c, t_r, t_c in TOPOLOGIES:
                print(f"  LIVE {detect}+{track}  detect {d_r}×{d_c}  track {t_r}×{t_c}",
                      flush=True)
                hota, note = e2e_hota(detect, track)
                summary = live_pair(detect, track, items, target, detect_cache[detect],
                                    d_r, d_c, t_r, t_c)
                rows.append(row_dict(detect, track, d_r, d_c, t_r, t_c, summary, hota, note,
                                     "live_detect", target, slo_ms))

    print("==> extra scale-out at 4 detect replicas (reference + best accuracy)", flush=True)
    for detect, track in (("yolo26n_imgsz640", "bytetrack"), ("yolo26n_imgsz1280", "ocsort")):
        print(f"  LIVE {detect}+{track}  detect 4×1", flush=True)
        hota, note = e2e_hota(detect, track)
        fn = detect_cache[detect]
        extra = [fn, load_detect_fn(detect, 1), load_detect_fn(detect, 1), load_detect_fn(detect, 1)]
        pin_threads(1)
        detect_pool = ReplicaPool.from_workers(
            [DetectWorker(detect, E2E_BATCH, 1, detect_fn=f) for f in extra], sticky=False)
        track_pool = make_track_pool(track, E2E_BATCH, 1, 1)

        def send(req, dp=detect_pool, tp=track_pool):
            det = dp.submit(req, req.stream_id)
            trk = tp.submit(
                {"stream_id": req.stream_id, "frame_id": req.frame_id, "boxes": det["boxes"]},
                req.stream_id,
            )
            return {
                "queue_s": (det.get("queue_s") or 0) + (trk.get("queue_s") or 0),
                "infer_s": (det.get("infer_s") or 0) + (trk.get("infer_s") or 0),
                "serdes_s": (det.get("serdes_s") or 0) + (trk.get("serdes_s") or 0),
            }

        summary = summarize(run_open_loop(send, items, target, workers=64))
        rows.append(row_dict(detect, track, 4, 1, 1, 1, summary, hota, note,
                             "live_detect", target, slo_ms))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "t7_e2e.csv", rows)
    (args.out / "t7_e2e.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out / 't7_e2e.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
