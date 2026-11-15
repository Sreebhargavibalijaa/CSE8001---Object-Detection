"""Open-loop load generator: send at a target rate whether or not replies have returned.

A closed-loop client (wait for each response) hides queueing and understates latency.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(records):
    ok = [row for row in records if row.get("ok")]
    lat = [row["latency_s"] for row in ok]
    queue = [row["queue_s"] for row in ok if row.get("queue_s") is not None]
    infer = [row["infer_s"] for row in ok if row.get("infer_s") is not None]
    serdes = [row["serdes_s"] for row in ok if row.get("serdes_s") is not None]
    elapsed = (records[-1]["done"] - records[0]["scheduled"]) if records else 0.0
    achieved = len(ok) / elapsed if elapsed > 0 else 0.0
    return {
        "n": len(records),
        "ok": len(ok),
        "errors": len(records) - len(ok),
        "p50_s": percentile(lat, 50),
        "p95_s": percentile(lat, 95),
        "p99_s": percentile(lat, 99),
        "p50_ms": None if not lat else percentile(lat, 50) * 1000,
        "p95_ms": None if not lat else percentile(lat, 95) * 1000,
        "p99_ms": None if not lat else percentile(lat, 99) * 1000,
        "queue_p50_ms": None if not queue else percentile(queue, 50) * 1000,
        "queue_p95_ms": None if not queue else percentile(queue, 95) * 1000,
        "infer_p50_ms": None if not infer else percentile(infer, 50) * 1000,
        "infer_p95_ms": None if not infer else percentile(infer, 95) * 1000,
        "serdes_p50_ms": None if not serdes else percentile(serdes, 50) * 1000,
        "serdes_p95_ms": None if not serdes else percentile(serdes, 95) * 1000,
        "achieved_rps": achieved,
        "elapsed_s": elapsed,
    }


def _meets(summary, slo_s):
    return (
        summary["ok"] == summary["n"]
        and summary["p95_s"] is not None
        and summary["p95_s"] <= slo_s
    )


def run_open_loop(send, items, rate_rps, workers=256):
    """Fire `send(item)` at `rate_rps` without waiting for previous replies."""
    if rate_rps <= 0:
        raise ValueError("rate must be positive")
    records = [None] * len(items)

    def one(index, item, scheduled):
        wait = scheduled - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        started = time.perf_counter()
        try:
            payload = send(item)
            done = time.perf_counter()
            row = {
                "ok": True,
                "scheduled": scheduled,
                "started": started,
                "done": done,
                "latency_s": done - started,
                "late_start_s": max(0.0, started - scheduled),
            }
            if isinstance(payload, dict):
                row.update({k: payload[k] for k in ("queue_s", "infer_s", "serdes_s", "result")
                            if k in payload})
        except Exception as error:
            done = time.perf_counter()
            row = {
                "ok": False,
                "scheduled": scheduled,
                "started": started,
                "done": done,
                "latency_s": done - started,
                "error": str(error),
            }
        records[index] = row

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(8, min(workers, max(8, len(items))))) as pool:
        futures = []
        for index, item in enumerate(items):
            scheduled = t0 + index / float(rate_rps)
            futures.append(pool.submit(one, index, item, scheduled))
        for future in futures:
            future.result()
    return [row for row in records if row is not None]


def max_rate_meeting_slo(send, items, slo_s, low=1.0, high=None, rounds=4, workers=256):
    """Highest open-loop rate whose p95 latency is ≤ slo_s. Returns (rate, summary).

    Characterizes service time with a short burst (does not wait for replies), then
    binary-searches around 1 / infer_p50 so low-rate probes do not dominate wall time.
    Never probes below 1 rps: 48 frames at 0.25 rps would sleep for minutes.
    """
    burst_rate = 16.0 if high is None else min(32.0, max(8.0, float(high)))
    burst = summarize(run_open_loop(send, items, burst_rate, workers=workers))
    burst["target_rps"] = burst_rate
    infer_s = (burst.get("infer_p50_ms") or burst.get("p50_ms") or 500.0) / 1000.0
    infer_p95_s = (burst.get("infer_p95_ms") or burst.get("p95_ms") or infer_s * 1000) / 1000.0
    est = 1.0 / max(infer_s, 0.02)
    lo = max(1.0, float(low))
    hi = float(high) if high is not None else max(est * 1.8, lo * 4, 2.0)
    print(f"    burst {burst_rate:.1f} rps  infer_p50 {infer_s*1000:.0f} ms  "
          f"p95 {None if burst['p95_s'] is None else burst['p95_s']*1000:.0f} ms  "
          f"search [{lo:.2f}, {hi:.2f}]", flush=True)

    # If inference alone already exceeds the SLO, no open-loop rate can pass.
    if infer_p95_s > slo_s:
        print(f"    infer p95 {infer_p95_s*1000:.0f} ms > SLO {slo_s*1000:.0f} ms → max rps 0",
              flush=True)
        return 0.0, burst

    best_rate, best_summary = 0.0, None
    if _meets(burst, slo_s) and burst_rate <= hi:
        best_rate, best_summary = burst_rate, burst

    for _ in range(rounds):
        mid = (lo + hi) / 2.0
        summary = summarize(run_open_loop(send, items, mid, workers=workers))
        summary["target_rps"] = mid
        ok = _meets(summary, slo_s)
        print(f"    rate {mid:.2f} rps  p95 {None if summary['p95_s'] is None else summary['p95_s']*1000:.0f} ms  "
              f"ok {summary['ok']}/{summary['n']}  {'PASS' if ok else 'FAIL'}", flush=True)
        if ok:
            best_rate, best_summary = mid, summary
            lo = mid
        else:
            hi = mid
    if best_summary is None:
        quiet = min(hi, max(lo, 0.4 / max(infer_s, 0.02)))
        quiet = max(1.0, quiet)
        best_summary = summarize(run_open_loop(send, items, quiet, workers=workers))
        best_summary["target_rps"] = quiet
        best_rate = quiet if _meets(best_summary, slo_s) else 0.0
    return best_rate, best_summary
