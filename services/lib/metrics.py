"""Prometheus /metrics HTTP server for a pipeline stage."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class StageMetrics:
    def __init__(self, stage):
        self.stage = stage
        self.requests = Counter("mot_requests_total", "gRPC requests", ["stage", "status"])
        self.latency = Histogram("mot_request_latency_seconds", "Request latency",
                                 ["stage"], buckets=BUCKETS)
        self.queue_length = Gauge("mot_queue_length", "Requests waiting to be batched", ["stage"])
        self.batch_size = Gauge("mot_batch_size", "Last processed batch size", ["stage"])
        self.assertions = Counter("mot_assertion_total", "Assertion outcomes", ["stage", "result"])
        self.set_queue(0)
        self.set_batch(0)

    def observe(self, seconds, ok=True):
        self.latency.labels(stage=self.stage).observe(seconds)
        self.requests.labels(stage=self.stage, status="ok" if ok else "error").inc()

    def set_queue(self, value):
        self.queue_length.labels(stage=self.stage).set(value)

    def set_batch(self, value):
        self.batch_size.labels(stage=self.stage).set(value)

    def assertion(self, passed, failed=0):
        if passed:
            self.assertions.labels(stage=self.stage, result="pass").inc(passed)
        if failed:
            self.assertions.labels(stage=self.stage, result="fail").inc(failed)


def serve_metrics(port):
    start_http_server(int(port))
