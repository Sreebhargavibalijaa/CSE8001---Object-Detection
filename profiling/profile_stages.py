"""T6: 4 variants × 4 resource configs × batch {1,4,8} per scalable stage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mot_pipeline (Object_Detection + Tracking)")]

from loadgen.open_loop import max_rate_meeting_slo, run_open_loop, summarize  # noqa: E402
from loadgen.pool import cached_detections, stream_frames  # noqa: E402
from profiling.accuracy_lookup import detect_hota, track_hota  # noqa: E402
from profiling.runtime import (  # noqa: E402
    BATCH_SIZES, DETECT_VARIANTS, RESOURCE_CONFIGS, TRACK_VARIANTS,
    DetectWorker, ReplicaPool, load_detect_fn, make_track_pool, pin_threads,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(send, items, slo_s, rounds):
    max_rps, slo_summary = max_rate_meeting_slo(
        send, items, slo_s, low=1.0, high=None, rounds=rounds, workers=64)
    probe = slo_summary if slo_summary is not None else summarize(
        run_open_loop(send, items, 1.0, workers=64))
    return probe, max_rps, slo_summary


def bottleneck_note(probe):
    parts = {
        "queue": probe.get("queue_p95_ms") or 0,
        "infer": probe.get("infer_p95_ms") or 0,
        "serdes": probe.get("serdes_p95_ms") or 0,
    }
    dominant = max(parts, key=parts.get)
    return (
        f"p95 split queue={parts['queue']:.0f}ms infer={parts['infer']:.0f}ms "
        f"serdes={parts['serdes']:.0f}ms; dominant={dominant}"
    )


def profile_detect(items, slo_s, rounds):
    rows = []
    for variant in DETECT_VARIANTS:
        print(f"\n=== detect variant {variant} ===", flush=True)
        fn_a = load_detect_fn(variant, cores=1)
        fn_b = None
        hota, hota_note = detect_hota(variant)
        for resource in RESOURCE_CONFIGS:
            pin_threads(resource["cores"])
            if resource["replicas"] >= 2 and fn_b is None:
                fn_b = load_detect_fn(variant, resource["cores"])
            fns = [fn_a] if resource["replicas"] == 1 else [fn_a, fn_b]
            for batch in BATCH_SIZES:
                print(f"  replicas={resource['replicas']} cores={resource['cores']} batch={batch}",
                      flush=True)
                workers = [DetectWorker(variant, batch, resource["cores"], detect_fn=fn)
                           for fn in fns]
                pool = ReplicaPool.from_workers(workers, sticky=False)
                send = lambda req, p=pool: p.submit(req, req.stream_id)
                probe, max_rps, _ = measure(send, items, slo_s, rounds)
                cost = resource["replicas"] * resource["cores"]
                rows.append({
                    "stage": "detect",
                    "variant": variant,
                    "batch": batch,
                    "replicas": resource["replicas"],
                    "cores_per_replica": resource["cores"],
                    "cost": cost,
                    "p50_ms": probe["p50_ms"],
                    "p95_ms": probe["p95_ms"],
                    "p99_ms": probe["p99_ms"],
                    "queue_p95_ms": probe["queue_p95_ms"],
                    "infer_p95_ms": probe["infer_p95_ms"],
                    "serdes_p95_ms": probe["serdes_p95_ms"],
                    "max_throughput_slo_rps": max_rps,
                    "accuracy_hota": hota,
                    "accuracy_note": hota_note,
                    "errors": probe["errors"],
                    "slo_p95_ms": slo_s * 1000,
                    "notes": bottleneck_note(probe),
                })
    return rows


def profile_track(items, slo_s, rounds):
    cached = cached_detections("yolo26n_imgsz640")
    rows = []
    for variant in TRACK_VARIANTS:
        print(f"\n=== track variant {variant} ===", flush=True)
        hota, hota_note = track_hota(variant)
        for resource in RESOURCE_CONFIGS:
            for batch in BATCH_SIZES:
                print(f"  replicas={resource['replicas']} cores={resource['cores']} batch={batch}",
                      flush=True)
                pool = make_track_pool(variant, batch, resource["replicas"], resource["cores"])

                def send(req, p=pool, dets=cached):
                    boxes = dets.get(req.sequence, {}).get(req.frame_id, [])
                    return p.submit(
                        {"stream_id": req.stream_id, "frame_id": req.frame_id, "boxes": boxes},
                        req.stream_id,
                    )

                probe, max_rps, _ = measure(send, items, slo_s, rounds)
                cost = resource["replicas"] * resource["cores"]
                rows.append({
                    "stage": "track",
                    "variant": variant,
                    "batch": batch,
                    "replicas": resource["replicas"],
                    "cores_per_replica": resource["cores"],
                    "cost": cost,
                    "p50_ms": probe["p50_ms"],
                    "p95_ms": probe["p95_ms"],
                    "p99_ms": probe["p99_ms"],
                    "queue_p95_ms": probe["queue_p95_ms"],
                    "infer_p95_ms": probe["infer_p95_ms"],
                    "serdes_p95_ms": probe["serdes_p95_ms"],
                    "max_throughput_slo_rps": max_rps,
                    "accuracy_hota": hota,
                    "accuracy_note": hota_note,
                    "errors": probe["errors"],
                    "slo_p95_ms": slo_s * 1000,
                    "notes": bottleneck_note(probe) + "; upstream detections cached (yolo26n_imgsz640)",
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sla", type=Path, default=ROOT / "docs" / "sla.json")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "profile")
    parser.add_argument("--stage", choices=["detect", "track", "both"], default="both")
    parser.add_argument("--replays", type=int, default=2)
    args = parser.parse_args()
    sla = json.loads(args.sla.read_text(encoding="utf-8"))
    slo_s = float(sla["slo_p95_s"])
    items = stream_frames(max_frames=args.frames, replays=args.replays)
    print(f"T6 n={len(items)} SLO={slo_s*1000:.0f} ms", flush=True)
    rows = []
    if args.stage in ("detect", "both"):
        rows.extend(profile_detect(items, slo_s, args.rounds))
    if args.stage in ("track", "both"):
        rows.extend(profile_track(items, slo_s, args.rounds))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "t6_stage_profile.csv", rows)
    (args.out / "t6_stage_profile.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out / 't6_stage_profile.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
