"""Parse Prometheus text format from a stage /metrics endpoint into JSON."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+(\S+)')
PLAIN = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\S+)')


def labels(blob):
    out = {}
    for part in blob.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


def parse(text):
    series = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if match:
            series[match.group(1)].append((labels(match.group(2)), float(match.group(3))))
            continue
        match = PLAIN.match(line)
        if match:
            series[match.group(1)].append(({}, float(match.group(2))))
    return series


def quantile(histogram, q):
    buckets = sorted(
        ((float(labels.get("le", "inf").replace("+Inf", "inf")), value)
         for labels, value in histogram if "le" in labels),
        key=lambda item: item[0] if item[0] != float("inf") else 1e18,
    )
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    target = q * total
    for bound, count in buckets:
        if count >= target:
            return bound * 1000 if bound != float("inf") else None
    return None


def summarize(text, stage_hint=None):
    series = parse(text)
    stage = stage_hint
    requests = 0
    for labels, value in series.get("mot_requests_total", []):
        stage = stage or labels.get("stage")
        requests += value
    queue = next((v for _, v in series.get("mot_queue_length", [])), 0)
    batch = next((v for _, v in series.get("mot_batch_size", [])), 0)
    passed = sum(v for labels, v in series.get("mot_assertion_total", []) if labels.get("result") == "pass")
    failed = sum(v for labels, v in series.get("mot_assertion_total", []) if labels.get("result") == "fail")
    total_a = passed + failed
    hist = series.get("mot_request_latency_seconds_bucket", [])
    return {
        "requests": int(requests),
        "p50_ms": quantile(hist, 0.50),
        "p95_ms": quantile(hist, 0.95),
        "p99_ms": quantile(hist, 0.99),
        "queue_length": queue,
        "batch_size": batch,
        "assertion_pass": int(passed),
        "assertion_fail": int(failed),
        "pass_rate": f"{(100.0 * passed / total_a):.1f}%" if total_a else "n/a",
    }


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def main():
    mapping = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "detect": "http://127.0.0.1:8081/metrics",
        "track": "http://127.0.0.1:8082/metrics",
        "evaluate": "http://127.0.0.1:8083/metrics",
    }
    stages = {}
    for stage, url in mapping.items():
        try:
            stages[stage] = summarize(fetch(url), stage)
        except Exception as error:
            stages[stage] = {"error": str(error)}
    print(json.dumps({"stages": stages}, indent=2))


if __name__ == "__main__":
    main()
