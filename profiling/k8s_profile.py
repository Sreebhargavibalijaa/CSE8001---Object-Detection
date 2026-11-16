"""T5-T7 measured on the real Kubernetes deployment (k3s on the course VM).

Every measurement point is set up in code, never by hand:

  kubectl patch    MODEL_NAME, BATCH_SIZE, CPU request = limit = cores per replica,
                   OMP threads = cores, number of replicas
  kubectl rollout  wait until every pod of the stage is Ready with that config

Then an open-loop load generator (loadgen.open_loop) sends labelled test-slice
frames over gRPC straight to the stage pods. Detection is round-robin across pods;
tracker frames are routed by stream_id (same md5 sharding as the gateway), and
every probe uses fresh stream ids so each replay starts a new tracker.

Latency is measured by the client, end to end per request. Each server returns its
queue time and inference time in gRPC trailing metadata; network + serialization
is the remainder. Cost = replicas x cores per replica.

  python -m profiling.k8s_profile preflight        # every variant loads, answers, is not throttled
  python -m profiling.k8s_profile calibrate        # T5: L0, SLO = 2 x L0, T0, target = 4 x T0
  python -m profiling.k8s_profile stages --resume  # T6: per-stage sweep
  python -m profiling.k8s_profile e2e --resume     # T7: end-to-end configs + gold input
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
sys.path[:0] = [str(ROOT), str(PIPELINE)]

import grpc  # noqa: E402

from loadgen.open_loop import percentile, run_open_loop, summarize  # noqa: E402
from loadgen.pool import cached_detections, stream_frames  # noqa: E402
from services import mot_pb2, mot_pb2_grpc  # noqa: E402
from services.lib.routing import tracker_shard  # noqa: E402

NAMESPACE = "mot17"
DETECT_VARIANTS = (
    "yolo26n-pt-imgsz640",
    "yolo26n-pt-imgsz1280",
    "yolo26n-onnx-imgsz640",
    "yolo26n-onnx-int8-imgsz640",
    "fasterrcnn-mv3-320",
)
TRACK_VARIANTS = ("bytetrack", "bytetrack-low", "bytetrack-long", "ocsort")
RESOURCES = ((1, 1), (2, 1), (4, 1), (1, 2))  # (replicas, cores per replica)
BATCHES = (1, 4, 8)
REFERENCE = {"detect": "yolo26n-pt-imgsz640", "track": "bytetrack", "batch": 1,
             "replicas": 1, "cores": 1}
TRACK_UPSTREAM = "yolo26n-pt-imgsz640"  # cached detections fed to the tracker in T6
STAGES = {"detect": ("deployment", "detect"), "track": ("statefulset", "track")}
MEMORY = {"detect": ("1Gi", "3Gi"), "track": ("512Mi", "1Gi")}
GRPC_OPTIONS = [("grpc.max_send_message_length", 32 << 20),
                ("grpc.max_receive_message_length", 32 << 20)]
ACCURACY_CSV = PIPELINE / "results" / "guide_eval_vm" / "test_summary.csv"

T6_FIELDS = [
    "stage", "variant", "batch", "replicas", "cores_per_replica", "cost",
    "p50_ms", "p95_ms", "p99_ms", "queue_p95_ms", "infer_p95_ms", "serdes_p95_ms",
    "max_throughput_slo_rps", "accuracy_hota", "accuracy_note", "errors", "slo_p95_ms",
    "notes", "low_load_p50_ms", "low_load_p95_ms", "rate_capped", "requests_at_max",
    "slo_attainment_pct", "cpu_throttled_pct", "measured_on", "measured_at",
]
T7_FIELDS = [
    "detect", "track", "load_multiplier", "target_rps", "batch", "detect_batch", "track_batch",
    "detect_replicas", "detect_cores", "track_replicas", "track_cores", "replicas",
    "cores_per_replica", "cost", "p50_ms", "p95_ms", "p99_ms", "queue_p95_ms",
    "infer_p95_ms", "serdes_p95_ms", "meets_sla", "slo_attainment_pct", "achieved_rps",
    "core_seconds", "elapsed_s", "requests", "errors", "assertion_pass_rate_under_load",
    "hota", "mota", "idf1", "accuracy_note", "source", "slo_p95_ms", "notes",
    "measured_on", "measured_at",
]


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# --------------------------------------------------------------------------- kubectl

def kubectl(*args, capture=True):
    result = subprocess.run(["kubectl", "-n", NAMESPACE, *args], text=True,
                            capture_output=capture, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed:\n{result.stderr or result.stdout}")
    return result.stdout


class Cluster:
    """Applies one (variant, batch, replicas, cores) setting to a stage, in code."""

    def __init__(self):
        self.applied = {}

    def apply(self, stage, variant, batch, replicas, cores):
        kind, name = STAGES[stage]
        wanted = (variant, int(batch), int(replicas), int(cores))
        if self.applied.get(stage) != wanted:
            cpu = str(int(cores))
            mem_request, mem_limit = MEMORY[stage]
            spec_extra = {}
            if kind == "deployment":
                # Old pods must be gone before new ones start: no overlap, no CPU overcommit.
                spec_extra["strategy"] = {"type": "Recreate", "rollingUpdate": None}
            patch = {"spec": spec_extra | {
                "replicas": int(replicas),
                "template": {
                    "metadata": {"annotations": {
                        "profiling/config": f"{variant}|b{batch}|c{cores}"}},
                    "spec": {"containers": [{
                        "name": name,
                        "env": [
                            {"name": "MODEL_NAME", "value": variant},
                            {"name": "BATCH_SIZE", "value": str(batch)},
                            {"name": "OMP_NUM_THREADS", "value": cpu},
                            {"name": "MKL_NUM_THREADS", "value": cpu},
                            {"name": "OPENBLAS_NUM_THREADS", "value": cpu},
                            {"name": "GRPC_WORKERS", "value": "64"},
                        ],
                        "resources": {
                            "requests": {"cpu": cpu, "memory": mem_request},
                            "limits": {"cpu": cpu, "memory": mem_limit},
                        },
                    }]},
                },
            }}
            log(f"  kubectl patch {kind}/{name}: {variant} batch={batch} "
                f"replicas={replicas} cores={cores}")
            kubectl("patch", f"{kind}/{name}", "--type", "strategic", "-p", json.dumps(patch))
            kubectl("rollout", "status", f"{kind}/{name}", "--timeout=900s")
            self.applied[stage] = wanted
        return self.ready_pods(stage, int(replicas))

    def cpu_stat(self, pods):
        """Summed cgroup CPU usage and throttled time (microseconds) of the given pods."""
        usage = throttled = 0
        try:
            for name, _ in pods:
                text = kubectl("exec", name, "--", "cat", "/sys/fs/cgroup/cpu.stat")
                values = dict(line.split()[:2] for line in text.splitlines() if line.strip())
                usage += int(values.get("usage_usec", 0))
                throttled += int(values.get("throttled_usec", 0))
        except Exception:
            return None
        return usage, throttled

    @staticmethod
    def throttled_pct(before, after):
        """Throttled time as a percentage of CPU time used between two cpu_stat() calls."""
        if not before or not after or after[0] <= before[0]:
            return None
        return 100.0 * (after[1] - before[1]) / (after[0] - before[0])

    def ready_pods(self, stage, replicas, timeout=900):
        deadline = time.time() + timeout
        while True:
            data = json.loads(kubectl("get", "pods", "-l", f"app={stage}", "-o", "json"))
            live, ready = [], []
            for pod in data["items"]:
                if pod["metadata"].get("deletionTimestamp"):
                    continue
                live.append(pod)
                conditions = {c["type"]: c["status"] for c in pod["status"].get("conditions", [])}
                if pod["status"].get("phase") == "Running" and conditions.get("Ready") == "True":
                    ready.append((pod["metadata"]["name"], pod["status"]["podIP"]))
            if len(live) == replicas and len(ready) == replicas:
                if stage == "track":  # StatefulSet: order by ordinal for stream sharding
                    ready.sort(key=lambda item: int(item[0].rsplit("-", 1)[1]))
                return ready
            if time.time() > deadline:
                raise TimeoutError(f"{stage}: {len(ready)}/{replicas} pods ready after {timeout}s")
            time.sleep(3)


# --------------------------------------------------------------------------- clients

def trailing_times(call):
    meta = dict(call.trailing_metadata() or ())
    return float(meta.get("x-queue-ms", 0.0)) / 1000.0, float(meta.get("x-infer-ms", 0.0)) / 1000.0


class DetectClient:
    def __init__(self, pods):
        self.channels = [grpc.insecure_channel(f"{ip}:50051", options=GRPC_OPTIONS) for _, ip in pods]
        self.stubs = [mot_pb2_grpc.DetectorStub(channel) for channel in self.channels]
        self.counter = itertools.count()

    def detect(self, frame, timeout=120):
        stub = self.stubs[next(self.counter) % len(self.stubs)]
        response, call = stub.Detect.with_call(mot_pb2.DetectRequest(frames=[frame]), timeout=timeout)
        queue_s, infer_s = trailing_times(call)
        return list(response.detections[0].boxes), queue_s, infer_s

    def warm(self, frame, per_pod=3):
        for stub in self.stubs:
            for _ in range(per_pod):
                stub.Detect(mot_pb2.DetectRequest(frames=[frame]), timeout=300)

    def close(self):
        for channel in self.channels:
            channel.close()


class TrackClient:
    def __init__(self, pods):
        self.channels = [grpc.insecure_channel(f"{ip}:50051", options=GRPC_OPTIONS) for _, ip in pods]
        self.stubs = [mot_pb2_grpc.TrackerStub(channel) for channel in self.channels]

    def track(self, request, timeout=120):
        shard, _ = tracker_shard(request.stream_id, len(self.stubs))
        response, call = self.stubs[shard].Track.with_call(request, timeout=timeout)
        queue_s, infer_s = trailing_times(call)
        return response, queue_s, infer_s

    def warm(self):
        for index, stub in enumerate(self.stubs):
            stub.Track(mot_pb2.TrackRequest(stream_id=f"warmup-{index}-{time.time()}", frame_id=1,
                                            width=1920, height=1080), timeout=120)

    def close(self):
        for channel in self.channels:
            channel.close()


# --------------------------------------------------------------------------- load pool

class LoadPool:
    """Test-slice frames as interleaved streams; extra replays get new stream ids."""

    def __init__(self, frames_per_sequence):
        self.frames_per_sequence = frames_per_sequence
        self.jpeg = {}
        self.lock = threading.Lock()
        self.base = len(stream_frames(max_frames=frames_per_sequence, replays=1))

    def take(self, count, rate=None):
        """`count` requests as concurrent streams of about 30 FPS each.

        Load = concurrent streams x frames/s (guide), so at `rate` requests/s we run
        max(3, ceil(rate / 30)) streams side by side. Every stream is one test-slice
        sequence replay with its own stream id; when a stream ends, the next replay
        continues in the same lane.
        """
        lanes = max(3, math.ceil((rate or 0) / 30.0))
        per_stream = self.frames_per_sequence
        streams_needed = max(lanes, math.ceil(count / per_stream) + lanes)
        replays = max(1, math.ceil(streams_needed / 3))
        items = stream_frames(max_frames=self.frames_per_sequence, replays=replays)
        streams = {}
        for item in items:
            streams.setdefault(item.stream_id, []).append(item)
        lane_items = [[] for _ in range(lanes)]
        for index, frames in enumerate(streams.values()):
            lane_items[index % lanes].extend(frames)
        out = []
        for position in range(max(len(lane) for lane in lane_items)):
            for lane in lane_items:
                if position < len(lane):
                    out.append(lane[position])
            if len(out) >= count:
                break
        return out[:count]

    def jpeg_bytes(self, path):
        data = self.jpeg.get(path)
        if data is None:
            data = Path(path).read_bytes()  # MOT17 frames are already JPEG
            with self.lock:
                self.jpeg[path] = data
        return data

    def frame(self, item, stream_id):
        return mot_pb2.Frame(stream_id=stream_id, frame_id=item.frame_id,
                             jpeg=self.jpeg_bytes(item.path), width=item.width, height=item.height)


def boxes_proto(rows):
    return [mot_pb2.Box(x1=r[0], y1=r[1], x2=r[2], y2=r[3], confidence=r[4]) for r in rows]


PROBE_COUNTER = itertools.count(int(time.time()) % 100000)


def detect_sender(pool, client):
    def factory():
        suffix = f"@p{next(PROBE_COUNTER)}"

        def send(item):
            t0 = time.perf_counter()
            _, queue_s, infer_s = client.detect(pool.frame(item, item.stream_id + suffix))
            total = time.perf_counter() - t0
            return {"queue_s": queue_s, "infer_s": infer_s,
                    "serdes_s": max(0.0, total - queue_s - infer_s)}
        return send
    return factory


def track_sender(boxes_by_sequence, client):
    def factory():
        suffix = f"@p{next(PROBE_COUNTER)}"

        def send(item):
            rows = boxes_by_sequence.get(item.sequence, {}).get(item.frame_id, [])
            request = mot_pb2.TrackRequest(stream_id=item.stream_id + suffix, frame_id=item.frame_id,
                                           width=item.width, height=item.height,
                                           boxes=boxes_proto(rows))
            t0 = time.perf_counter()
            response, queue_s, infer_s = client.track(request)
            total = time.perf_counter() - t0
            return {"queue_s": queue_s, "infer_s": infer_s,
                    "serdes_s": max(0.0, total - queue_s - infer_s),
                    "result": (response.assertions_passed, response.assertions_failed)}
        return send
    return factory


def e2e_sender(pool, detect_client, track_client):
    def factory():
        suffix = f"@p{next(PROBE_COUNTER)}"

        def send(item):
            stream_id = item.stream_id + suffix
            t0 = time.perf_counter()
            boxes, q1, i1 = detect_client.detect(pool.frame(item, stream_id))
            response, q2, i2 = track_client.track(mot_pb2.TrackRequest(
                stream_id=stream_id, frame_id=item.frame_id, width=item.width,
                height=item.height, boxes=boxes))
            total = time.perf_counter() - t0
            return {"queue_s": q1 + q2, "infer_s": i1 + i2,
                    "serdes_s": max(0.0, total - q1 - q2 - i1 - i2),
                    "result": (response.assertions_passed, response.assertions_failed)}
        return send
    return factory


# --------------------------------------------------------------------------- probes

class Prober:
    def __init__(self, pool, args, slo_s=None):
        self.pool = pool
        self.args = args
        self.slo_s = slo_s

    def run(self, factory, rate, count=None):
        count = count or int(min(self.args.max_requests,
                                 max(self.args.min_requests, round(rate * self.args.seconds))))
        items = self.pool.take(count, rate)
        records = run_open_loop(factory(), items, rate, workers=self.args.workers)
        for row in records:
            # Open loop: latency counts from the scheduled send time, so time spent
            # waiting for a free client thread is not hidden. That wait is queueing.
            late = max(0.0, row["started"] - row["scheduled"])
            row["latency_s"] = row["done"] - row["scheduled"]
            if row.get("queue_s") is not None:
                row["queue_s"] += late
        summary = summarize(records)
        summary["target_rps"] = rate
        late = [row.get("late_start_s", 0.0) for row in records if row.get("ok")]
        summary["late_start_p95_ms"] = None if not late else percentile(late, 95) * 1000
        if self.slo_s is not None:
            within = sum(1 for row in records if row.get("ok") and row["latency_s"] <= self.slo_s)
            summary["slo_attainment_pct"] = 100.0 * within / max(1, len(records))
        results = [row["result"] for row in records if row.get("ok") and row.get("result")]
        if results:
            passed = sum(p for p, _ in results)
            failed = sum(f for _, f in results)
            summary["assertion_pass_rate"] = 100.0 * passed / max(1, passed + failed)
        errors = [row.get("error") for row in records if not row.get("ok")]
        if errors:
            summary["first_error"] = errors[0]
        return summary

    def service_time(self, factory, count=4):
        """Median latency of single requests sent one at a time (no queueing at all)."""
        send = factory()
        times = []
        for item in self.pool.take(count):
            started = time.perf_counter()
            send(item)
            times.append(time.perf_counter() - started)
        times.sort()
        return times[len(times) // 2]

    def low_rate(self, factory, replicas, utilization):
        """A rate that keeps every replica mostly idle: "very low load"."""
        service_s = self.service_time(factory)
        rate = min(self.args.low_rate, utilization * max(1, replicas) / max(service_s, 1e-3))
        return max(rate, 0.02), service_s

    def meets(self, summary):
        return (summary["errors"] == 0 and summary["p95_s"] is not None
                and summary["p95_s"] <= self.slo_s
                and summary["achieved_rps"] >= 0.85 * summary["target_rps"])

    def max_rate(self, factory, replicas):
        """Highest open-loop rate whose p95 <= SLO (no errors). Returns (rate, at_rate, low, capped)."""
        args = self.args
        low_rate, single_s = self.low_rate(factory, replicas, args.low_utilization)
        low = self.run(factory, low_rate, count=args.low_requests)
        log(f"    single request {single_s * 1000:.0f} ms; low load {low_rate:.2f} rps: "
            f"p50 {fmt(low['p50_ms'])} p95 {fmt(low['p95_ms'])} ms errors {low['errors']}")
        if not self.meets(low):
            return 0.0, low, low, False
        service_s = max((low.get("infer_p50_ms") or low["p50_ms"] or 1.0) / 1000.0, 1e-3)
        best_rate, best = low_rate, low
        hi = min(args.max_rate, max(low_rate * 2, 1.2 * replicas / service_s))
        failed_at = None
        for _ in range(8):  # grow until a probe fails (or the generator cap)
            summary = self.run(factory, hi)
            ok = self.meets(summary)
            log(f"    {hi:7.1f} rps: p95 {fmt(summary['p95_ms'])} ms  "
                f"errors {summary['errors']}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failed_at = hi
                break
            best_rate, best = hi, summary
            if hi >= args.max_rate:
                return best_rate, best, low, True
            hi = min(args.max_rate, hi * 2)
        if failed_at is None:
            return best_rate, best, low, True
        lo, hi = best_rate, failed_at
        for _ in range(args.rounds):  # geometric bisection between last pass and first fail
            if hi / lo < 1.08:
                break
            mid = math.sqrt(lo * hi)
            summary = self.run(factory, mid)
            ok = self.meets(summary)
            log(f"    {mid:7.1f} rps: p95 {fmt(summary['p95_ms'])} ms  "
                f"errors {summary['errors']}  {'PASS' if ok else 'FAIL'}")
            if ok:
                lo, best_rate, best = mid, mid, summary
            else:
                hi = mid
        return best_rate, best, low, False


def fmt(value):
    return "-" if value is None else f"{value:.0f}"


# --------------------------------------------------------------------------- accuracy + files

def load_accuracy():
    if not ACCURACY_CSV.is_file():
        raise FileNotFoundError(f"{ACCURACY_CSV} missing: run guide_offline.py first")
    with ACCURACY_CSV.open(newline="", encoding="utf-8") as handle:
        return {(row["detector"], row["tracker"]): row for row in csv.DictReader(handle)}


def acc_value(accuracy, detector, tracker, key="HOTA"):
    row = accuracy.get((detector, tracker))
    if not row or row.get(key) in (None, "", "None"):
        return None
    return float(row[key])


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(path, fields, row):
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def measured_on():
    try:
        nodes = json.loads(subprocess.run(["kubectl", "get", "nodes", "-o", "json"], text=True,
                                          capture_output=True, check=True).stdout)
        info = nodes["items"][0]["status"]
        cpus = info["capacity"]["cpu"]
        return f"k3s node {nodes['items'][0]['metadata']['name']} ({cpus} vCPU), CPU only"
    except Exception:
        return "k3s on course VM, CPU only"


def load_sla(path):
    sla = json.loads(Path(path).read_text(encoding="utf-8"))
    if "k8s" not in sla.get("method", ""):
        raise RuntimeError(f"{path} was not produced by k8s_profile calibrate; run that first")
    return sla


def breakdown_note(summary):
    parts = {"queue": summary.get("queue_p95_ms") or 0.0,
             "inference": summary.get("infer_p95_ms") or 0.0,
             "network/serdes": summary.get("serdes_p95_ms") or 0.0}
    dominant = max(parts, key=parts.get)
    return (f"p95 split queue={parts['queue']:.0f}ms inference={parts['inference']:.0f}ms "
            f"network/serdes={parts['network/serdes']:.0f}ms; dominant={dominant}")


# --------------------------------------------------------------------------- preflight

def preflight(args):
    """Load every variant once in a 1-core pod and check it answers, unthrottled.

    Takes about 10 minutes and stops the run early if any variant fails to load,
    returns errors, or runs more threads than its CPU limit (heavy CFS throttling).
    """
    pool = LoadPool(args.frames_per_sequence)
    cluster = Cluster()
    upstream = cached_detections(TRACK_UPSTREAM)
    if not any(upstream.values()):
        raise FileNotFoundError(f"no cached {TRACK_UPSTREAM} test-slice detections; run guide_offline.py")
    load_accuracy()
    checks = [("detect", v, 1) for v in DETECT_VARIANTS] + [("detect", REFERENCE["detect"], 2)]
    checks += [("track", v, 1) for v in TRACK_VARIANTS]
    rows, failures = [], []
    for stage, variant, cores in checks:
        log(f"preflight {stage} {variant} (1 replica x {cores} core)")
        row = {"stage": stage, "variant": variant, "cores": cores}
        try:
            pods = cluster.apply(stage, variant, 1, 1, cores)
            if stage == "detect":
                client = DetectClient(pods)
                client.warm(pool.frame(pool.take(1)[0], "warmup"))
                factory = detect_sender(pool, client)
            else:
                client = TrackClient(pods)
                client.warm()
                factory = track_sender(upstream, client)
            prober = Prober(pool, args)
            single_s = prober.service_time(factory, count=6)
            rate = min(100.0, 0.5 / max(single_s, 1e-3))
            before = cluster.cpu_stat(pods)
            summary = prober.run(factory, rate, count=int(max(15, rate * 6)))
            throttled = cluster.throttled_pct(before, cluster.cpu_stat(pods))
            client.close()
            row.update({"single_request_ms": single_s * 1000, "probe_rps": rate,
                        "p95_ms": summary["p95_ms"], "errors": summary["errors"],
                        "cpu_throttled_pct": throttled})
            problem = None
            if summary["errors"]:
                problem = f"{summary['errors']} errors: {summary.get('first_error', '')[:160]}"
            elif throttled is not None and throttled > args.max_throttled_pct:
                problem = (f"CPU throttled {throttled:.0f}% of its CPU time at half load "
                           "(more threads than cores?)")
            log(f"    single request {single_s * 1000:.0f} ms, p95 at half load "
                f"{fmt(summary['p95_ms'])} ms, throttled {fmt(throttled)}%  "
                f"{'OK' if problem is None else 'PROBLEM: ' + problem}")
        except Exception as error:
            cluster.applied.pop(stage, None)
            problem = f"{type(error).__name__}: {' '.join(str(error).split())[:300]}"
            log(f"    PROBLEM: {problem}")
        row["problem"] = problem
        rows.append(row)
        if problem:
            failures.append(f"{stage} {variant}: {problem}")
    one = next((r for r in rows if r["stage"] == "detect" and r["variant"] == REFERENCE["detect"]
                and r["cores"] == 1), {})
    two = next((r for r in rows if r["stage"] == "detect" and r["cores"] == 2), {})
    if one.get("single_request_ms") and two.get("single_request_ms"):
        log(f"reference detector: {one['single_request_ms']:.0f} ms on 1 core, "
            f"{two['single_request_ms']:.0f} ms on 2 cores")
    out = Path(args.out) / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measured_on": measured_on(), "checks": rows,
                               "failures": failures}, indent=2, default=str), encoding="utf-8")
    if failures:
        log("PREFLIGHT FAILED:")
        for failure in failures:
            log(f"  - {failure}")
        raise SystemExit(1)
    log(f"PREFLIGHT PASSED ({len(rows)} checks) -> {out}")


# --------------------------------------------------------------------------- T5 calibrate

def calibrate(args):
    cluster = Cluster()
    pool = LoadPool(args.frames_per_sequence)
    ref = REFERENCE
    log(f"T5 calibration with reference {ref} (1 replica x 1 core per stage)")
    detect_pods = cluster.apply("detect", ref["detect"], ref["batch"], 1, 1)
    track_pods = cluster.apply("track", ref["track"], ref["batch"], 1, 1)
    dclient, tclient = DetectClient(detect_pods), TrackClient(track_pods)
    try:
        dclient.warm(pool.frame(pool.take(1)[0], "warmup"))
        tclient.warm()
        prober = Prober(pool, args)
        factory = e2e_sender(pool, dclient, tclient)
        calib_rate, single_s = prober.low_rate(factory, 1, args.calib_utilization)
        log(f"single request {single_s * 1000:.0f} ms -> L0 probe: {args.calib_requests} requests "
            f"at {calib_rate:.3f} rps (~{args.calib_utilization:.0%} utilization, very low load)")
        low = prober.run(factory, calib_rate, count=args.calib_requests)
        if low["errors"]:
            raise RuntimeError(f"low-load probe had errors: {low.get('first_error')}")
        l0_s = low["p95_s"]
        slo_s = 2.0 * l0_s
        prober.slo_s = slo_s
        log(f"L0 p95 = {l0_s * 1000:.1f} ms  ->  SLO p95 = {slo_s * 1000:.1f} ms")
        t0, at_t0, _, capped = prober.max_rate(factory, replicas=1)
        target = 4.0 * t0
        log(f"T0 = {t0:.1f} frames/s  ->  target = {target:.1f} frames/s")
    finally:
        dclient.close()
        tclient.close()
    sla = {
        "guide": "Pipeline Evaluation Guide v4, common SLA calibration procedure",
        "method": "k8s: open-loop gRPC load against the deployed detect + track pods",
        "hardware": measured_on(),
        "reference": ref | {"cores_per_replica": 1, "replicas_per_stage": 1},
        "request": "one frame of one MOT17 test-slice stream (detect -> track)",
        "unit": "p95 per-frame latency (ms) and sustained frames/s; streams = frames/s / 30",
        "native_fps": 30,
        "L0_requests": args.calib_requests,
        "L0_rate_rps": calib_rate,
        "single_request_ms": single_s * 1000,
        "L0_p95_ms": l0_s * 1000,
        "slo_p95_ms": slo_s * 1000,
        "slo_p95_s": slo_s,
        "T0_rps": t0,
        "T0_streams_at_30fps": t0 / 30.0,
        "T0_capped_by_generator": capped,
        "target_rps": target,
        "target_streams_at_30fps": target / 30.0,
        "multiplier": 4,
        "low_load": low,
        "T0_summary": at_t0,
        "load_pool": ("eval_slices/MOT17_test: second halves of MOT17-02/04/09-FRCNN, "
                      f"first {args.frames_per_sequence} frames each, streams interleaved "
                      "round-robin; replays get new stream ids"),
        "trace": "deterministic frame order, no random sampling (seed not needed)",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(args.sla)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sla, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out}")
    return sla


# --------------------------------------------------------------------------- T6 stages

def stage_points(stage):
    variants = DETECT_VARIANTS if stage == "detect" else TRACK_VARIANTS
    for variant in variants:
        for batch in BATCHES:
            for cores in sorted({c for _, c in RESOURCES}):
                for replicas in sorted(r for r, c in RESOURCES if c == cores):
                    yield variant, batch, replicas, cores


def profile_stages(args):
    sla = load_sla(args.sla)
    slo_s = float(sla["slo_p95_s"])
    accuracy = load_accuracy()
    out = Path(args.out) / "t6_stage_profile.csv"
    stale = any(abs(float(r["slo_p95_ms"] or 0) - slo_s * 1000) > 0.5 for r in read_rows(out))
    if out.is_file() and (not args.resume or stale):
        if stale:
            log("existing T6 rows were measured with a different SLO; starting T6 fresh")
        out.rename(out.with_suffix(f".{datetime.now():%Y%m%d_%H%M%S}.bak"))
    done = {(r["stage"], r["variant"], r["batch"], r["replicas"], r["cores_per_replica"])
            for r in read_rows(out)}
    pool = LoadPool(args.frames_per_sequence)
    cluster = Cluster()
    where = measured_on()
    stages = ("detect", "track") if args.stage == "both" else (args.stage,)
    points = [(stage, *p) for stage in stages for p in stage_points(stage)]
    upstream = cached_detections(TRACK_UPSTREAM) if "track" in stages else None
    if upstream is not None and not any(upstream.values()):
        raise FileNotFoundError(f"no cached {TRACK_UPSTREAM} test-slice detections; run guide_offline.py")
    started = time.time()
    for index, (stage, variant, batch, replicas, cores) in enumerate(points, 1):
        key = (stage, variant, str(batch), str(replicas), str(cores))
        if key in done:
            continue
        elapsed = time.time() - started
        log(f"T6 [{index}/{len(points)}] {stage} {variant} batch={batch} "
            f"replicas={replicas} cores={cores}   (elapsed {elapsed / 60:.0f} min)")
        hota, acc_note, throttled = None, "", None
        try:
            if stage == "detect":
                pods = cluster.apply("detect", variant, batch, replicas, cores)
                client = DetectClient(pods)
                client.warm(pool.frame(pool.take(1)[0], "warmup"))
                factory = detect_sender(pool, client)
                hota = acc_value(accuracy, variant, "bytetrack")
                acc_note = "test-slice HOTA tracked with bytetrack"
            else:
                pods = cluster.apply("track", variant, batch, replicas, cores)
                client = TrackClient(pods)
                client.warm()
                factory = track_sender(upstream, client)
                hota = acc_value(accuracy, TRACK_UPSTREAM, variant)
                acc_note = f"test-slice HOTA on cached {TRACK_UPSTREAM} detections"
            prober = Prober(pool, args, slo_s)
            before = cluster.cpu_stat(pods)
            rate, at_rate, low, capped = prober.max_rate(factory, replicas)
            throttled = cluster.throttled_pct(before, cluster.cpu_stat(pods))
            client.close()
            note = breakdown_note(at_rate)
            if stage == "track":
                note += f"; upstream detections cached ({TRACK_UPSTREAM})"
            if capped:
                note += f"; reached generator cap {args.max_rate:.0f} rps"
            if rate == 0:
                note += "; p95 above SLO even at low load"
        except Exception as error:  # keep the sweep going overnight
            log(f"    ERROR {' '.join(str(error).split())[:400]}")
            cluster.applied.pop(stage, None)
            at_rate, low = {}, {}
            rate, capped = None, False
            note = f"ERROR: {str(error)[:200]}"
        append_row(out, T6_FIELDS, {
            "stage": stage, "variant": variant, "batch": batch, "replicas": replicas,
            "cores_per_replica": cores, "cost": replicas * cores,
            "p50_ms": at_rate.get("p50_ms"), "p95_ms": at_rate.get("p95_ms"),
            "p99_ms": at_rate.get("p99_ms"), "queue_p95_ms": at_rate.get("queue_p95_ms"),
            "infer_p95_ms": at_rate.get("infer_p95_ms"), "serdes_p95_ms": at_rate.get("serdes_p95_ms"),
            "max_throughput_slo_rps": rate, "accuracy_hota": hota, "accuracy_note": acc_note,
            "errors": at_rate.get("errors"), "slo_p95_ms": slo_s * 1000, "notes": note,
            "low_load_p50_ms": low.get("p50_ms"), "low_load_p95_ms": low.get("p95_ms"),
            "rate_capped": capped, "requests_at_max": at_rate.get("n"),
            "slo_attainment_pct": at_rate.get("slo_attainment_pct"),
            "cpu_throttled_pct": throttled,
            "measured_on": where, "measured_at": datetime.now().isoformat(timespec="seconds"),
        })
        log(f"    -> max throughput meeting SLO: {'-' if rate is None else f'{rate:.1f}'} rps, "
            f"p95 there {fmt(at_rate.get('p95_ms'))} ms")
    log(f"T6 done -> {out}")


# --------------------------------------------------------------------------- T7 end to end

def capacity_plan(t6, stage, variant, need_rps, max_replicas):
    """Cheapest (replicas, cores, batch) from T6 single-replica capacity, assuming linear scale-out."""
    best = None
    rows = [r for r in t6 if r["stage"] == stage and r["variant"] == variant
            and r["max_throughput_slo_rps"] not in ("", "None")]
    for cores in sorted({int(r["cores_per_replica"]) for r in rows}):
        by_batch = {}
        for r in rows:
            if int(r["cores_per_replica"]) != cores:
                continue
            per_replica = float(r["max_throughput_slo_rps"]) / int(r["replicas"])
            batch = int(r["batch"])
            # Worst measured per-replica capacity for this batch: scale-out is not perfectly linear.
            by_batch[batch] = min(by_batch.get(batch, per_replica), per_replica)
        if not by_batch:
            continue
        batch, capacity = max(by_batch.items(), key=lambda item: item[1])
        top = {"batch": batch}
        if capacity <= 0:
            continue
        replicas = min(max_replicas, max(1, math.ceil(need_rps * 1.2 / capacity)))
        plan = {"replicas": replicas, "cores": cores, "batch": int(top["batch"]),
                "capacity_per_replica": capacity, "cost": replicas * cores,
                "short": replicas * capacity < need_rps}
        if best is None or (plan["short"], plan["cost"]) < (best["short"], best["cost"]):
            best = plan
    return best


def profile_e2e(args):
    sla = load_sla(args.sla)
    slo_s = float(sla["slo_p95_s"])
    t0 = float(sla["T0_rps"])
    accuracy = load_accuracy()
    t6 = read_rows(Path(args.out) / "t6_stage_profile.csv")
    if not t6:
        raise FileNotFoundError("t6_stage_profile.csv missing: run the stages sweep first")
    out_all = Path(args.out) / "t7_e2e_all_loads.csv"
    stale = any(abs(float(r["slo_p95_ms"] or 0) - slo_s * 1000) > 0.5 for r in read_rows(out_all))
    if out_all.is_file() and (not args.resume or stale):
        if stale:
            log("existing T7 rows were measured with a different SLO; starting T7 fresh")
        out_all.rename(out_all.with_suffix(f".{datetime.now():%Y%m%d_%H%M%S}.bak"))
    done = {(r["detect"], r["track"], r["load_multiplier"]) for r in read_rows(out_all)}
    pool = LoadPool(args.frames_per_sequence)
    cluster = Cluster()
    where = measured_on()
    started = time.time()
    combos = [(m, d, t) for m in args.load_multipliers for d in DETECT_VARIANTS for t in TRACK_VARIANTS]
    for index, (multiplier, detect, track) in enumerate(combos, 1):
        key = (detect, track, str(multiplier))
        if key in done:
            continue
        need = multiplier * t0
        log(f"T7 [{index}/{len(combos)}] {detect} + {track} at {multiplier} x T0 = {need:.1f} rps "
            f"(elapsed {(time.time() - started) / 60:.0f} min)")
        d_plan = capacity_plan(t6, "detect", detect, need, args.max_detect_replicas)
        t_plan = capacity_plan(t6, "track", track, need, args.max_track_replicas)
        row = {
            "detect": detect, "track": track, "load_multiplier": multiplier, "target_rps": need,
            "hota": acc_value(accuracy, detect, track),
            "mota": acc_value(accuracy, detect, track, "MOTA"),
            "idf1": acc_value(accuracy, detect, track, "IDF1"),
            "accuracy_note": "Guide v4 test slice, TrackEval combined", "source": "live_detect",
            "slo_p95_ms": slo_s * 1000, "measured_on": where,
        }
        notes = []
        if d_plan is None or t_plan is None:
            notes.append("a stage cannot meet the SLO even at low load (from T6); not deployed")
            row.update({"meets_sla": False, "notes": "; ".join(notes),
                        "measured_at": datetime.now().isoformat(timespec="seconds")})
            append_row(out_all, T7_FIELDS, row)
            continue
        total = d_plan["cost"] + t_plan["cost"]
        if total > args.core_budget:
            spare = max(1, (args.core_budget - t_plan["cost"]) // d_plan["cores"])
            notes.append(f"detect replicas capped {d_plan['replicas']}->{spare} by core budget "
                         f"{args.core_budget}")
            d_plan["replicas"], d_plan["cost"] = spare, spare * d_plan["cores"]
        if d_plan["short"] or t_plan["short"]:
            notes.append("T6 capacity x replica cap is below the target rate")
        try:
            d_pods = cluster.apply("detect", detect, d_plan["batch"], d_plan["replicas"], d_plan["cores"])
            t_pods = cluster.apply("track", track, t_plan["batch"], t_plan["replicas"], t_plan["cores"])
            dclient, tclient = DetectClient(d_pods), TrackClient(t_pods)
            dclient.warm(pool.frame(pool.take(1)[0], "warmup"))
            tclient.warm()
            prober = Prober(pool, args, slo_s)
            count = int(max(args.min_requests, round(need * args.e2e_seconds)))
            summary = prober.run(e2e_sender(pool, dclient, tclient), need, count=count)
            meets = prober.meets(summary)
            for attempt in range(args.e2e_retries):
                # Missed the SLA: add detect replicas (the bottleneck stage) while cores remain.
                step = max(1, math.ceil(0.25 * d_plan["replicas"]))
                room = (args.core_budget - t_plan["cost"]) // d_plan["cores"] - d_plan["replicas"]
                step = min(step, room, args.max_detect_replicas - d_plan["replicas"])
                if meets or step < 1:
                    break
                dclient.close()
                d_plan["replicas"] += step
                d_plan["cost"] = d_plan["replicas"] * d_plan["cores"]
                log(f"    missed SLA (p95 {fmt(summary['p95_ms'])} ms); retry with "
                    f"{d_plan['replicas']} detect replicas")
                d_pods = cluster.apply("detect", detect, d_plan["batch"], d_plan["replicas"],
                                       d_plan["cores"])
                dclient = DetectClient(d_pods)
                dclient.warm(pool.frame(pool.take(1)[0], "warmup"))
                summary = prober.run(e2e_sender(pool, dclient, tclient), need, count=count)
                meets = prober.meets(summary)
                notes.append(f"scaled detect to {d_plan['replicas']} replicas after missing the SLA")
            dclient.close()
            tclient.close()
            cost = d_plan["cost"] + t_plan["cost"]
            notes.append(breakdown_note(summary))
            if summary.get("first_error"):
                notes.append(f"first error: {summary['first_error'][:120]}")
            row.update({
                "batch": f"{d_plan['batch']}/{t_plan['batch']}",
                "detect_batch": d_plan["batch"], "track_batch": t_plan["batch"],
                "detect_replicas": d_plan["replicas"], "detect_cores": d_plan["cores"],
                "track_replicas": t_plan["replicas"], "track_cores": t_plan["cores"],
                "replicas": d_plan["replicas"], "cores_per_replica": d_plan["cores"], "cost": cost,
                "p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"], "p99_ms": summary["p99_ms"],
                "queue_p95_ms": summary["queue_p95_ms"], "infer_p95_ms": summary["infer_p95_ms"],
                "serdes_p95_ms": summary["serdes_p95_ms"], "meets_sla": meets,
                "slo_attainment_pct": summary.get("slo_attainment_pct"),
                "achieved_rps": summary["achieved_rps"], "elapsed_s": summary["elapsed_s"],
                "core_seconds": cost * summary["elapsed_s"], "requests": summary["n"],
                "errors": summary["errors"],
                "assertion_pass_rate_under_load": summary.get("assertion_pass_rate"),
            })
            log(f"    cost {cost}  p95 {fmt(summary['p95_ms'])} ms  SLO {slo_s * 1000:.0f} ms  "
                f"{'MEETS' if meets else 'misses'} SLA  HOTA {row['hota']}")
        except Exception as error:
            log(f"    ERROR {error}")
            cluster.applied.clear()
            notes.append(f"ERROR: {str(error)[:200]}")
            row["meets_sla"] = False
        row["notes"] = "; ".join(notes)
        row["measured_at"] = datetime.now().isoformat(timespec="seconds")
        append_row(out_all, T7_FIELDS, row)

    rows = read_rows(out_all)
    target_rows = [r for r in rows if r["load_multiplier"] == str(max(args.load_multipliers))]
    for track in TRACK_VARIANTS:  # gold-input ablation (accuracy only; not a deployable config)
        target_rows.append({
            "detect": "gold_gt", "track": track, "source": "gold_input", "cost": 1,
            "hota": acc_value(accuracy, "gold_gt", track),
            "mota": acc_value(accuracy, "gold_gt", track, "MOTA"),
            "idf1": acc_value(accuracy, "gold_gt", track, "IDF1"),
            "accuracy_note": "gold-input ablation: tracker on ground-truth boxes (offline)",
            "meets_sla": "", "notes": "upper bound for the tracker; detector error removed",
        })
    target_rows.append({
        "detect": "public_frcnn", "track": "bytetrack", "source": "baseline_offline",
        "hota": acc_value(accuracy, "public_frcnn", "bytetrack"),
        "mota": acc_value(accuracy, "public_frcnn", "bytetrack", "MOTA"),
        "idf1": acc_value(accuracy, "public_frcnn", "bytetrack", "IDF1"),
        "accuracy_note": "guide baseline: ByteTrack on MOT17 public FRCNN detections (offline)",
        "meets_sla": "", "notes": "baseline",
    })
    out_target = Path(args.out) / "t7_e2e.csv"
    with out_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=T7_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(target_rows)
    write_t7_summary(rows, Path(args.out) / "t7_summary.json")
    log(f"T7 done -> {out_target} (target load) and {out_all} (all loads)")


def write_t7_summary(rows, path):
    """Numbers for the T7 questions: who meets the SLA, cheapest, best accuracy, per load."""
    summary = {}
    for multiplier in sorted({r["load_multiplier"] for r in rows}):
        live = [r for r in rows if r["load_multiplier"] == multiplier and r.get("cost") not in ("", None)]
        meeting = [r for r in live if r["meets_sla"] == "True"]
        best_tracker = {}
        for detect in DETECT_VARIANTS:
            options = [r for r in rows if r["detect"] == detect and r["load_multiplier"] == multiplier
                       and r.get("hota") not in ("", None)]
            if options:
                top = max(options, key=lambda r: float(r["hota"]))
                best_tracker[detect] = {"track": top["track"], "hota": float(top["hota"])}
        summary[f"{multiplier}x_T0"] = {
            "configs_run": len(live),
            "configs_meeting_sla": len(meeting),
            "cheapest_meeting_sla": None if not meeting else min(
                ({"detect": r["detect"], "track": r["track"], "cost": int(float(r["cost"])),
                  "hota": float(r["hota"]), "p95_ms": float(r["p95_ms"])} for r in meeting),
                key=lambda item: (item["cost"], -item["hota"])),
            "most_accurate_meeting_sla": None if not meeting else max(
                ({"detect": r["detect"], "track": r["track"], "cost": int(float(r["cost"])),
                  "hota": float(r["hota"]), "p95_ms": float(r["p95_ms"])} for r in meeting),
                key=lambda item: (item["hota"], -item["cost"])),
            "best_tracker_per_detector": best_tracker,
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {path}")


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="T5-T7 profiling on the k3s deployment")
    parser.add_argument("command", choices=["preflight", "calibrate", "stages", "e2e", "all"])
    parser.add_argument("--stage", choices=["detect", "track", "both"], default="both")
    parser.add_argument("--sla", default=str(ROOT / "docs" / "sla.json"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "profile"))
    parser.add_argument("--resume", action="store_true", help="skip points already in the CSV")
    parser.add_argument("--frames-per-sequence", type=int, default=120,
                        help="test-slice frames per sequence in the load pool (<= 263)")
    parser.add_argument("--seconds", type=float, default=12.0, help="length of each search probe")
    parser.add_argument("--min-requests", type=int, default=25)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--low-rate", type=float, default=1.0, help="upper bound for the low-load rate")
    parser.add_argument("--low-utilization", type=float, default=0.25,
                        help="low-load probe rate as a fraction of single-request capacity")
    parser.add_argument("--low-requests", type=int, default=15)
    parser.add_argument("--max-rate", type=float, default=400.0,
                        help="load-generator cap in requests/s (about 13 streams at 30 FPS)")
    parser.add_argument("--rounds", type=int, default=4, help="bisection rounds per point")
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--calib-utilization", type=float, default=0.10,
                        help="L0 probe rate as a fraction of single-request capacity")
    parser.add_argument("--calib-requests", type=int, default=40)
    parser.add_argument("--load-multipliers", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument("--e2e-seconds", type=float, default=30.0)
    parser.add_argument("--e2e-retries", type=int, default=3,
                        help="times to add detect replicas when an end-to-end config misses the SLA")
    parser.add_argument("--max-detect-replicas", type=int, default=10)
    parser.add_argument("--max-track-replicas", type=int, default=4)
    parser.add_argument("--core-budget", type=int, default=14,
                        help="max cores for detect+track pods together (VM has 16 vCPU)")
    parser.add_argument("--max-throttled-pct", type=float, default=35.0,
                        help="preflight fails above this CPU throttling at half load")
    args = parser.parse_args()
    if args.command in ("preflight", "all"):
        preflight(args)
    if args.command in ("calibrate", "all"):
        calibrate(args)
    if args.command in ("stages", "all"):
        profile_stages(args)
    if args.command in ("e2e", "all"):
        profile_e2e(args)


if __name__ == "__main__":
    main()
