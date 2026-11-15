"""Write one Word document with every Milestone 2 result."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
# Single deliverable. Older copies are overwritten with the same bytes.
OUT = ROOT / "CSE8001_Milestone2_All_Results.docx"


def shade(cell, color="1F4E79"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.bold = True
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(255, 255, 255)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = str(header)
        shade(table.rows[0].cells[i])
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            table.rows[r].cells[c].text = "" if value is None else str(value)
            for paragraph in table.rows[r].cells[c].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8)
    doc.add_paragraph("")
    return table


def heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def para(doc, text):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(11)
    return paragraph


def theory_para(doc, text):
    paragraph = para(doc, text)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.line_spacing = 1.35
    return paragraph


def add_theory(doc):
    """~6 pages of Milestone 2 theory. Placed first so the report can be read as a paper then as tables."""
    heading(doc, "Theory (read this first)")
    theory_para(
        doc,
        "This milestone is not “run YOLO and report FPS.” It is the study of a multi-stage "
        "serving system whose correctness is defined on labeled video, whose latency is defined "
        "under open-loop load, and whose cost is defined in replica-cores. The sections below "
        "are the theory needed to interpret every table that follows: why the pipeline is split "
        "into stages, why HOTA is the accuracy metric, why assertions are not a substitute for "
        "labels, why an SLA is a pair (p95 latency, target rate), why that pair is calibrated "
        "from a single replica, and why “the best configuration” is not a single point on a plot.",
    )

    heading(doc, "A. A pipeline, not a single model", 2)
    theory_para(
        doc,
        "Multi-object tracking (MOT) on MOT17 is a pipeline because the scientific questions "
        "and the systems questions attach to different cuts of the computation. Person detection "
        "turns a frame into a set of boxes. Association (ByteTrack or OC-SORT) turns a stream of "
        "boxes into persistent identities. Evaluation turns identities plus ground truth into HOTA, "
        "MOTA, IDF1, and online assertions. If detection and tracking were fused into one opaque "
        "binary, you could not ask how much error each stage contributes, you could not cache "
        "upstream outputs while profiling the tracker, and you could not scale the slow stage "
        "without scaling the fast one.",
    )
    theory_para(
        doc,
        "The natural unit of work is one frame of one video stream. MOT17 is recorded at 30 FPS, "
        "but 30 FPS is a property of the camera, not a promise the CPU stack can keep. A request "
        "in this report is therefore one frame; throughput is frames per second; latency is "
        "milliseconds per frame. Treating a whole video as one request would hide per-frame "
        "queueing and make the tracker’s intra-sequence state look like a batch job.",
    )
    theory_para(
        doc,
        "Detection is stateless: two frames can run on two replicas without talking to each other. "
        "Tracking is stateful: ByteTrack and OC-SORT keep Kalman filters, track IDs, and lost-track "
        "buffers per stream. Frames of one MOT17 sequence must therefore hit the same tracker "
        "replica, in frame-id order. That is why detect is a Deployment and track is a StatefulSet "
        "with sticky routing (hash of stream_id). Scale the tracker between videos, not mid-stream, "
        "or identities reset and HOTA collapses for a reason that has nothing to do with the model.",
    )
    theory_para(
        doc,
        "MOT17 ships each of seven train videos three times (DPM, FRCNN, SDP) with identical "
        "pixels and different public detections. Using all three copies would triple the frame "
        "count without adding information. Milestone 1 therefore uses one copy of each video. "
        "The Pipeline Evaluation Guide v4 further restricts comparable accuracy numbers to the "
        "static-camera FRCNN folders MOT17-02/04/09, split at half length: first half is "
        "development (tuning only), second half is test (everything reported below). Public MOT17 "
        "test has no ground truth, so all HOTA in this document is on train sequences under that "
        "split. Frame IDs in the trimmed slices are renumbered from 1.",
    )

    heading(doc, "B. Accuracy: HOTA, baselines, gold input, and assertions", 2)
    theory_para(
        doc,
        "HOTA (Higher Order Tracking Accuracy) is the primary metric because it jointly scores "
        "detection accuracy, association accuracy, and localization. MOTA is dominated by false "
        "positives and missed detections and can look healthy while IDs churn. IDF1 emphasises "
        "identity preservation. ID switches (IDSW) count identity breaks. Detector precision and "
        "recall at IoU 0.5, computed on boxes before the tracker, separate “the detector never "
        "saw the person” from “the tracker threw the box away.” TrackEval’s MOT17 pedestrian "
        "preprocessing (class 1, consider-flag 1) is applied so numbers match the MOT literature.",
    )
    theory_para(
        doc,
        "A published detector-tracker pair is meaningless without two reference points. The "
        "required baseline is ByteTrack with default thresholds on MOT17’s public FRCNN detections: "
        "that is a strong off-the-shelf point (HOTA 50.1 on our test slice) and it tells you whether "
        "your YOLO stack is actually better than the detections the benchmark already ships. The "
        "gold-input ablation feeds ground-truth pedestrian boxes to the tracker and skips the "
        "detector. If gold HOTA is ~97 and the full pipeline is ~47, almost all error is detection, "
        "not association. That single fact constrains every later design choice: swapping OC-SORT "
        "for ByteTrack cannot close a 50-point HOTA gap; raising YOLO imgsz can.",
    )
    theory_para(
        doc,
        "Online assertions (Guide A1–A3) are label-free checks on the tracker’s output. A1: a "
        "track’s centre must not jump more than half the current box height between consecutive "
        "frames (implausible motion). A2: height/width must lie in [1, 6] (implausible pedestrian "
        "shape). A3: the count of tracks must not swing by more than max(3, 0.3 × previous count). "
        "A frame passes iff A1 ∧ A2 ∧ A3. These are useful as serving health signals (Prometheus "
        "can scrape them with no GT), but they are not accuracy. On this slice they anti-correlate "
        "with HOTA (Spearman ρ ≈ −0.89) because A2 fails more often when recall is high: better "
        "detectors emit more truncated or squat boxes at image borders. A 100% assertion pass rate "
        "with 25% recall means the few tracks that exist look kinematic, not that tracking is good.",
    )
    theory_para(
        doc,
        "Milestone 1’s motion check (centre jump / image diagonal / frame ≤ 0.0157) was fit on all "
        "seven train videos, which include this test slice — it leaked. It is reported as "
        "m1_motion_pass_rate for continuity and is not used to rank configurations. Accuracy "
        "numbers that are comparable across later milestones are the Guide v4 combined HOTA on "
        "the 02/04/09 FRCNN second halves (1,088 frames), not the unweighted seven-sequence mean.",
    )

    heading(doc, "C. Variants, serving, batching, and why the image does not change", 2)
    theory_para(
        doc,
        "A serving system is profiled over variants because “YOLO” is not one operating point. "
        "Input resolution (imgsz 640 vs 1280) changes which pedestrians survive letterbox resize: "
        "MOT17 people are often tens of pixels tall in 1920×1080 frames; at 640 they occupy a "
        "handful of pixels and are dropped. Runtime (PyTorch vs ONNX Runtime vs dynamic INT8 on "
        "MatMul/Gemm) changes service time at nearly constant accuracy. Tracker family and "
        "thresholds (ByteTrack default, low-threshold, long buffer, OC-SORT) change association "
        "under occlusion. TensorRT is not used: the course VMs are CPU-only. Each scalable stage "
        "must expose at least four registered variants in MinIO so a live Deployment can switch "
        "MODEL_NAME without rebuilding the container.",
    )
    theory_para(
        doc,
        "gRPC plus a request batcher is the serving contract. The batcher waits until BATCH_SIZE "
        "or BATCH_TIMEOUT, then runs the handler. On a GPU, a true fused batch can raise "
        "throughput. On CPU YOLO the handler still runs images sequentially inside the batch, so "
        "batch 4 and 8 mostly add wait for the timeout under sparse arrivals and only help when "
        "open-loop bursts fill the queue. That is a measurement, not a slogan, and it is why the "
        "T6 table includes batch ∈ {1, 4, 8} rather than assuming “bigger batch is faster.”",
    )
    theory_para(
        doc,
        "Prometheus /metrics (p50/p95/p99 latency, throughput, queue length, CPU, assertion pass "
        "rate) is how a deployed pipeline is observed; Grafana is how a human reads it. Kubernetes "
        "requests and limits make those CPU numbers comparable: a pod that can steal eight cores "
        "on an idle node is not the same experiment as a pod pinned to one core. Scripted scale "
        "(kubectl scale via scale_stage.sh) is required because “we clicked 2 in the dashboard” "
        "is not a reproducible operation and because tracker replica count must stay consistent "
        "with the gateway’s hash modulus.",
    )

    heading(doc, "D. Anatomy of latency", 2)
    theory_para(
        doc,
        "End-to-end latency of a frame is not one number with one cause. We split it into three "
        "parts that the load generator records on every request. (1) Queueing time: waiting in "
        "the batcher or behind another in-flight batch on the same replica. This grows as offered "
        "load approaches capacity and is the term a closed-loop client refuses to measure. "
        "(2) Inference time: the model or tracker update itself. For YOLO this is a function of "
        "imgsz, runtime, and threads per replica; for ByteTrack/OC-SORT it is a function of how "
        "many boxes the detector emitted. (3) Network / serialization / decode time: reading the "
        "JPEG, packing protobufs, copying arrays. On this CPU pipeline decode is a few milliseconds; "
        "on a GPU server it can dominate if you are not careful. p50 describes the typical frame; "
        "p95 and p99 describe the tail the SLA cares about.",
    )
    theory_para(
        doc,
        "A stage is the bottleneck if its inference plus queueing dominate the sum, and if adding "
        "replicas there raises the maximum rate at which p95 still meets the SLO. In this pipeline "
        "that stage is detection. The tracker is a few milliseconds and saturates far above the "
        "detector’s capacity, so scaling track replicas does not fix a missed SLA at the target "
        "rate. More CPU threads on one replica help only if the implementation is parallel inside "
        "one predict(); YOLO26n on CPU is only weakly so, which is why 2 cores on 1 replica is "
        "not twice 1 core.",
    )

    heading(doc, "E. Why the load generator is open-loop", 2)
    theory_para(
        doc,
        "A closed-loop client sends frame i+1 only after frame i returns. The number of requests "
        "in the system is then bounded by the number of client threads (often 1). Queueing delay "
        "cannot grow, because the client stops offering work. Measured latency collapses toward "
        "service time, throughput is 1 / mean service time, and the system looks faster and more "
        "scalable than it is. That is the wrong experiment for a video pipeline: cameras do not "
        "wait. Frames arrive from the world at a rate that does not depend on whether last frame "
        "finished.",
    )
    theory_para(
        doc,
        "An open-loop generator schedules request i at t0 + i / λ, whether or not earlier "
        "requests have returned. If λ exceeds the service rate μ of the replicas, a queue builds "
        "and p95 explodes. That explosion is the signal that you must scale. Little’s law "
        "(L = λW) is the accounting identity behind it: at a given rate, more wait means more "
        "items in system. Percentile latency at open-loop λ is also the right way to talk about "
        "an SLA, because the SLA is a promise under an offered load, not under a polite client.",
    )
    theory_para(
        doc,
        "Requests are drawn from the labeled load pool — the Guide test-slice frames — so that "
        "accuracy can be scored under load on the same GT used offline. Inventing synthetic boxes "
        "would let you measure latency and not know whether the variant that is fast is also the "
        "variant that is wrong. Exhausted pool cycles with a new stream_id so the tracker’s "
        "in-memory state resets and TrackEval still has GT for that replay.",
    )

    heading(doc, "F. Why an SLA, and how it is calibrated", 2)
    theory_para(
        doc,
        "An SLA in this course is not “as fast as possible” and not “30 FPS because the camera "
        "says so.” It is a contract with two numbers: a latency objective (p95, in the pipeline’s "
        "natural unit, here milliseconds per frame) and a target request rate (frames/s). You "
        "need both. A latency bound with no rate is trivial (send one frame a minute). A rate "
        "with no latency bound is trivial (queue forever, drain later). Together they say: at "
        "this offered load, the 95th percentile frame still finishes in time.",
    )
    theory_para(
        doc,
        "p95 is used instead of the mean because the mean hides tails. A detector that is 20 ms "
        "usually and 2 s occasionally has a fine mean and a broken tracker: late boxes arrive "
        "after the person has moved, IDs break, and HOTA falls for a latency reason that MOTA "
        "will not explain. p99 is reported as a diagnostic; the objective itself is p95 so that "
        "a few warmup outliers do not make the contract impossible on CPU.",
    )
    theory_para(
        doc,
        "The Guide v4 calibration procedure exists so that teams cannot pick an SLA their single "
        "replica already meets. Procedure: (1) Choose a reference configuration — here YOLO26n "
        "imgsz=640 + ByteTrack, batch 1, 1 replica × 1 CPU core. (2) Warm up so cold-start compile "
        "is not L0. (3) Measure L0 = p95 latency at very low load, where queueing is ~0 and L0 "
        "is essentially service time. (4) Set the latency SLO to 2 × L0. The factor two is "
        "headroom: some queueing is allowed, but not a collapse. (5) Search for T0, the highest "
        "open-loop rate at which p95 still ≤ SLO on that one replica. (6) Set the SLA target rate "
        "to 4 × T0. Meeting 4 × T0 on the reference hardware requires more than one replica "
        "(roughly four if scaling is linear; more if it is not). That is the point. An SLA you "
        "can meet without scaling does not exercise T4–T6.",
    )
    theory_para(
        doc,
        "The SLO is inherited by every other variant. YOLO-1280 has a larger service time, so it "
        "may already sit near 2 × L0_640 at low load and will need many more replicas to hit the "
        "same target rate. That is not unfair; it is the trade-off the Pareto plot is for. You "
        "may not silently switch the SLA to “whatever 1280 can do” when the question is which "
        "configuration meets a stated contract.",
    )
    theory_para(
        doc,
        "CPU-only, explicit cores per replica, is mandatory. GPU or multi-process speedups may be "
        "reported in addition, clearly labeled, but not instead. A number that depends on an idle "
        "Mac using eight AMX cores is not the same experiment as a VM with one core in the limit. "
        "Cost is defined as replicas × CPU cores per replica (summed across stages for end-to-end). "
        "That definition makes “add a replica” and “give the replica more cores” comparable on one "
        "axis.",
    )

    heading(doc, "G. Profiling, caching, Pareto, and what “optimal” means", 2)
    theory_para(
        doc,
        "T6 sweeps each scalable stage over at least four variants × four resource configurations "
        "× batch {1, 4, 8}. For each point we report p50/p95/p99, max open-loop throughput whose "
        "p95 still meets the SLO, and accuracy. Accuracy is a function of the variant (and, for "
        "some pipelines, of batch-dependent input knobs). It does not change with replica count. "
        "If it appears to, the measurement is wrong: you are either scoring different frames, "
        "resetting tracker state inconsistently, or leaking GT. When profiling the tracker, "
        "upstream detections are computed once per detector variant and reused. Re-running YOLO "
        "for every ByteTrack YAML would confound tracker latency with detector latency and waste "
        "the only expensive resource in the pipeline.",
    )
    theory_para(
        doc,
        "T7 combines stages. The full detect × track cross-product is scored for accuracy on the "
        "test slice. A subset is run live at the SLA target rate on more than one topology so that "
        "cost and p95 are real. Gold-input rows on the same axes show the accuracy ceiling if the "
        "detector were perfect. The cost–accuracy plot marks which live points meet the SLA and "
        "draws the Pareto front of deployable points: no other live config is both cheaper and "
        "more accurate. Gold is not on that front; it is an ablation, not a detector you can serve.",
    )
    theory_para(
        doc,
        "Does more than one configuration meet the SLA? At 4 × T0, possibly none of the 1–4 "
        "replica live points do — that is evidence the target is doing its job. At T0, several "
        "cheap 640-class variants do. Optimal is therefore load-dependent. If the contract is the "
        "SLA target rate, you deploy the fastest accurate-enough detector with enough replicas "
        "that T6’s max-SLO-throughput ≥ target (here, YOLO-640 + ByteTrack at five 1-core "
        "replicas, cost 6, HOTA 36.6). If the offered load drops to T0 and HOTA is the objective, "
        "YOLO-1280 + OC-SORT at one replica is the live Pareto pick (HOTA 47.0) even though it "
        "cannot hold 153 frames/s. The upstream detector does change the best tracker (640 prefers "
        "ByteTrack, 1280 prefers OC-SORT) but by ~1 HOTA, which is small next to the +10 HOTA from "
        "imgsz. There is no single “best” row independent of λ.",
    )
    theory_para(
        doc,
        "Monitoring is part of the same theory, not a screenshot requirement. A histogram of "
        "request latency is the only way to read p50/p95/p99 after the fact; a counter of "
        "completed requests is the only way to read throughput; a gauge of queue length is the "
        "only way to see Little’s L without reconstructing it from traces. Assertion pass rate "
        "on /metrics is the online stand-in for “is this replica producing implausible boxes,” "
        "which is why it appears on the Grafana dashboard next to CPU. None of those time series "
        "replace TrackEval. They tell you whether the SLA is being missed because of queueing, "
        "because of a slow variant, or because a replica is wedged.",
    )
    theory_para(
        doc,
        "The rest of this document is the evidence for those claims: harness numbers on the Guide "
        "slice, the variant catalog, the Kubernetes live scrape, the calibrated SLA, the T6 "
        "stage tables, the T7 configuration table, and the plots. Read the theory as the "
        "definitions; read the tables as the measurements those definitions require.",
    )
    br = doc.add_paragraph()
    br.add_run().add_break(WD_BREAK.PAGE)


def caption(doc, text):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    return paragraph


def add_picture(doc, path, cap=None):
    path = Path(path)
    if not path.is_file():
        return
    doc.add_picture(str(path), width=Inches(6.4))
    if cap:
        caption(doc, cap)


def fmt(value, digits=1):
    if value in (None, "", "None"):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build():
    catalog = json.loads((PIPELINE / "registry" / "catalog.json").read_text(encoding="utf-8"))
    summary = read_csv(PIPELINE / "results" / "summary.csv")
    per_seq_m1 = read_csv(PIPELINE / "results" / "per_sequence.csv")
    det_recall = read_csv(PIPELINE / "results" / "detection_recall.csv")
    det_size = read_csv(PIPELINE / "results" / "detection_recall_by_size.csv")
    guide = json.loads((PIPELINE / "results" / "guide_eval" / "test_report.json").read_text(encoding="utf-8"))
    guide_seq = read_csv(PIPELINE / "results" / "guide_eval" / "test_per_sequence.csv")
    live = json.loads((ROOT / "docs" / "live_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "eval_slices" / "manifest.json").read_text(encoding="utf-8"))
    version_txt = (ROOT / "eval_slices" / "DATASET_VERSION.txt").read_text(encoding="utf-8").strip()

    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("CSE 8001 Milestone 2 — all results")
    run.bold = True
    run.font.size = Pt(22)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("MOT17 detection + tracking  ·  Guide v4 harness, T0–T7")
    r.font.size = Pt(13)
    para(doc, "This is the single results document. It opens with the theory (pipeline, accuracy, "
         "open-loop load, why an SLA, cost and Pareto), then every measured number from "
         "the evaluation harness (Guide v4 test slice), Milestone 1 seven-sequence profiling, "
         "the MinIO variant registry, the gRPC stage servers, Kubernetes + Prometheus/Grafana, "
         "the open-loop load generator, SLA calibration, per-stage profiling, and end-to-end Pareto.")

    add_theory(doc)

    sla_path = ROOT / "docs" / "sla.json"
    t6_path = ROOT / "docs" / "profile" / "t6_stage_profile.csv"
    t7_path = ROOT / "docs" / "profile" / "t7_e2e.csv"
    sla = json.loads(sla_path.read_text(encoding="utf-8")) if sla_path.is_file() else None
    t6 = read_csv(t6_path) if t6_path.is_file() else []
    t7 = read_csv(t7_path) if t7_path.is_file() else []

    heading(doc, "Results at a glance")
    if sla:
        add_table(doc, ["result", "value"], [
            ["SLA (p95 latency objective)", f"{sla['slo_p95_ms']:.1f} ms / frame"],
            ["SLA target rate", f"{sla['target_rps']:.2f} frames/s  (4 × T0)"],
            ["L0 p95 at low load (1×1 YOLO-640 + ByteTrack)", f"{sla['L0_p95_ms']:.1f} ms"],
            ["T0 max rate meeting SLO on 1 replica", f"{sla['T0_rps']:.2f} frames/s"],
            ["Guide test-slice gold-input HOTA", "97.3"],
            ["Best YOLO pipeline HOTA (imgsz=1280 + OC-SORT)", "47.0"],
            ["Public FRCNN + ByteTrack baseline HOTA", "50.1"],
            ["Chosen M1 config HOTA on guide slice", "46.7 (YOLO-1280 + ByteTrack)"],
            ["Deploy at SLA target", "YOLO-640 + ByteTrack, 5 detect replicas × 1 core, cost 6, HOTA 36.6"],
            ["Bottleneck", "detect inference + queueing; tracker is a few ms"],
            ["T6 rows / T7 rows", f"{len(t6)} / {len(t7)}"],
        ])
    para(doc, "Full tables and plots for T5–T7 are in sections 10–13. Open-loop load on the labeled "
         "Guide v4 test slice (MOT17-02/04/09-FRCNN second halves). CPU-only; cores/replica = torch/OMP threads.")

    heading(doc, "Contents")
    for item in (
        "Theory A–G (pipeline, accuracy, serving, latency, open-loop, SLA, Pareto)",
        "Results at a glance",
        "1. Pipeline and hardware",
        "2. Evaluation harness (Guide v4 slices, metrics, assertions)",
        "3. Guide test-slice results (combined, per sequence, baseline, gold-input)",
        "4. Assertion pass rate vs HOTA",
        "5. Milestone 1 accuracy definition vs the guide",
        "6. T1 seven-sequence profiling (M1 definition)",
        "7. T2 MinIO registry variants",
        "8. T3 gRPC stage servers and sticky tracker routing",
        "9. T4 Kubernetes, Prometheus, Grafana, live run",
        "10. T5 Open-loop load generator and SLA",
        "11. T6 Per-stage profiling (CPU, explicit cores/replica)",
        "12. T7 End-to-end configurations, gold ablation, Pareto",
        "13. Deploy recommendation at the SLA target rate",
        "14. How to change a variant without rebuilding",
        "15. Presentation outline",
    ):
        para(doc, item)

    heading(doc, "1. Pipeline and hardware")
    para(doc, "Stages: person detection → multi-object tracking → TrackEval + online assertions. "
         "Detectors: YOLO26n (imgsz 640 / 960 / 1280) and Faster R-CNN MobileNetV3-Large 320-FPN. "
         "The torchvision Faster R-CNN ResNet-50 FPN was dropped (~0.13 FPS on CPU; a 5,316-frame sweep is impractical). "
         "Trackers: ByteTrack (default and low-threshold), OC-SORT. "
         "Everything is pretrained. No training. CPU-only VMs — TensorRT is not used. "
         "Graph export is ONNX Runtime; quantization is ONNX Runtime dynamic INT8 on MatMul/Gemm only.")
    add_table(doc, ["item", "value"], [
        ["Dataset", "MOT17 train (public test has no GT)"],
        ["Guide folders", "MOT17-02/04/09-FRCNN (static cameras)"],
        ["M1 folders", "seven unique videos, DPM copy (frames identical to FRCNN/SDP)"],
        ["Primary offline metric (guide)", "TrackEval HOTA, MOT17 pedestrian preprocessing, combined"],
        ["Also", "MOTA, IDF1, IDSW, detector P/R at IoU 0.5"],
        ["Baseline", "ByteTrack defaults on MOT17 public FRCNN detections"],
        ["Gold-input ablation", "ByteTrack on GT pedestrian boxes (class 1, consider-flag 1)"],
        ["Chosen M1 config", "YOLO26n imgsz=1280 + ByteTrack"],
        ["Serving image", "mot17-serve:local (variant selected by MODEL_NAME)"],
        ["Cluster", "Docker Desktop Kubernetes v1.36.1, namespace mot17"],
    ])

    heading(doc, "2. Evaluation harness (Pipeline Evaluation Guide v4)")
    para(doc, "Accuracy numbers are comparable only on the guide slices. "
         "Development slice is for tuning; results below are test-slice only. "
         "A1–A3 thresholds are written in the guide (no t-grid). Milestone 1 knobs were not re-tuned on test. "
         "numpy.random.default_rng(8001) is recorded; the video split is deterministic half-length, not a random sample.")
    para(doc, version_txt.replace("\n", " | "))
    add_table(doc, ["slice", "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN", "trimmed frames"], [
        ["development (tune only)", "orig 1–300 → 1–300", "orig 1–525 → 1–525", "orig 1–262 → 1–262", "1087"],
        ["test (reported)", "orig 301–600 → 1–300", "orig 526–1050 → 1–525", "orig 263–525 → 1–263", "1088"],
    ])
    hash_rows = []
    for name, info in manifest["sequences"].items():
        hash_rows.append([
            name, info["seq_length"], int(info["frame_rate"]),
            f"{info['width']}×{info['height']}",
            info["seqinfo_sha256"][:16], info["gt_sha256"][:16], info["det_sha256"][:16],
        ])
    add_table(doc, ["sequence", "seqLength", "fps", "size", "seqinfo sha256", "gt sha256", "det sha256"], hash_rows)
    para(doc, "Required assertions (per frame; the frame passes iff A1 ∧ A2 ∧ A3). "
         "A1 plausible motion: for every track in this frame and the previous one, centre distance ≤ 0.5 × current box height. "
         "A2 plausible shape: height/width in [1.0, 6.0]. "
         "A3 stable count: |n_t − n_{t−1}| ≤ max(3, 0.3 × previous count).")
    para(doc, "Extra A4 (ours): every tracked box overlaps the image rectangle. "
         "Justification: a box with no image overlap cannot be a visible pedestrian; the check needs no labels. "
         "Milestone 1 motion (centre jump / image diagonal / frame ≤ 0.0157, fit on all seven train videos) is still reported as m1_motion_pass_rate. "
         "That threshold was fit on data that includes this test slice — it leaked, and is not a development-slice threshold.")

    heading(doc, "3. Guide test-slice results")
    para(doc, "Gold-input HOTA 97.3 vs best full pipeline 47.0 (YOLO26n imgsz=1280 + OC-SORT) means almost all error is the detector. "
         "The required baseline (public FRCNN detections + ByteTrack) is HOTA 50.1, above our YOLO pipeline on this slice.")
    heading(doc, "Combined across MOT17-02/04/09 FRCNN second halves (1088 frames)", 2)
    add_table(doc, ["detector", "tracker", "HOTA", "MOTA", "IDF1", "IDSW", "Prec", "Rec", "det P", "det R",
                    "A1–A3", "A1", "A2", "A3", "A4", "M1 motion"], [
        [row["detector"], row["tracker"],
         fmt(row["HOTA"]), fmt(row["MOTA"]), fmt(row["IDF1"]), row["IDSW"],
         fmt(row.get("Precision")), fmt(row.get("Recall")),
         fmt(row.get("det_precision")), fmt(row.get("det_recall")),
         fmt(row["assertion_pass_rate"]), fmt(row["A1_pass_rate"]), fmt(row["A2_pass_rate"]),
         fmt(row["A3_pass_rate"]), fmt(row["A4_pass_rate"]), fmt(row["m1_motion_pass_rate"])]
        for row in guide["combined"]
    ])
    heading(doc, "Per sequence on the test slice", 2)
    add_table(doc, ["sequence", "detector", "tracker", "HOTA", "MOTA", "IDF1", "IDSW", "Rec",
                    "A1–A3", "A2", "M1 motion", "det P", "det R"], [
        [row["sequence"], row["detector"], row["tracker"],
         fmt(row["HOTA"]), fmt(row["MOTA"]), fmt(row["IDF1"]), row["IDSW"], fmt(row["Recall"]),
         fmt(row["assertion_pass_rate"]), fmt(row["A2_pass_rate"]), fmt(row["m1_motion_pass_rate"]),
         fmt(row.get("det_precision")), fmt(row.get("det_recall"))]
        for row in guide_seq
    ])

    heading(doc, "4. Does assertion pass rate track HOTA?")
    ranking = guide["ranking"]
    para(doc, f"No. On the 12 Milestone 1 configs, Spearman ρ(HOTA, A1∧A2∧A3) = {ranking['spearman_hota_vs_A1A2A3']:.2f} "
         f"(Pearson {ranking['pearson_hota_vs_A1A2A3']:.2f}). They do not rank configurations the same way. "
         "A1 never fails (100% on every config). A3 almost never fails. A2 (aspect ratio) is what moves the pass rate: "
         "higher-recall detectors emit more squat or truncated boxes with height/width < 1, so the best HOTA configs "
         "have the worst assertion pass rates. "
         f"Milestone 1 motion is 99.2–99.5% for every config (Spearman vs HOTA = {ranking['spearman_hota_vs_m1_motion']:.2f}) "
         "and also does not rank them.")
    add_table(doc, ["rank", "by HOTA (best first)", "by A1–A3 pass rate (best first)"], [
        [str(i), h, a] for i, (h, a) in enumerate(zip(ranking["hota_order"], ranking["assertion_order"]), start=1)
    ])

    heading(doc, "5. Milestone 1 accuracy definition vs the guide")
    add_table(doc, ["", "Milestone 1 (as submitted)", "Guide v4 (this harness)"], [
        ["Sequences", "02/04/05/09/10/11/13, one DPM copy, full length", "02/04/09 FRCNN, second half only"],
        ["Frames", "5,316", "1,088 (renumbered from 1)"],
        ["HOTA aggregation", "unweighted mean of per-sequence HOTA", "TrackEval combine_sequences"],
        ["Assertion", "per track-step, centre / image diagonal ≤ 0.0157", "per frame, A1 ∧ A2 ∧ A3"],
        ["Threshold source", "99th percentile on all 7 train videos (includes test)", "fixed in the guide"],
        ["Baseline", "not reported", "public FRCNN dets + ByteTrack"],
        ["Gold-input", "not reported", "ByteTrack on GT pedestrians"],
        ["Chosen config HOTA", "41.3 (YOLO 1280 + ByteTrack, 7-seq mean)", "46.7 same config on test slice; OC-SORT 47.0"],
        ["Best non-gold on guide slice", "—", "public FRCNN + ByteTrack HOTA 50.1"],
    ])
    para(doc, "Both definitions are reported in this document. The seven-sequence table is historical Milestone 1; "
         "comparable numbers for later milestones are the Guide v4 test-slice table in section 3.")

    heading(doc, "6. T1 seven-sequence profiling (Milestone 1 definition)")
    para(doc, "YOLO26n at imgsz=640 looked like ~90% precision / ~31–43% TrackEval recall because letterboxing shrinks "
         "far-field pedestrians. Raising imgsz to 1280 is the real lever (mean recall 43% → 55%). "
         "ByteTrack-low changes TrackEval recall by about one point. "
         "Faster R-CNN MobileNetV3-320 is the CPU-practical two-stage baseline (~16 FPS, HOTA 23), replacing ResNet-50 FPN. "
         "M1 motion pass is 96.9–97.5% for every config, including Faster R-CNN at HOTA 23.")
    heading(doc, "Configuration means (unweighted over 7 sequences)", 2)
    add_table(doc, ["detector", "tracker", "HOTA", "MOTA", "IDF1", "Recall", "Precision", "M1 motion", "failed/checked"], [
        [row["detector"], row["tracker"], fmt(row["HOTA"]), fmt(row["MOTA"]), fmt(row["IDF1"]),
         fmt(row["Recall"]), fmt(row["Precision"]), fmt(row["assertion_pass_rate"]),
         f"{row['assertions_failed']}/{row['assertions_checked']}"]
        for row in summary
    ])
    heading(doc, "Per-sequence HOTA and M1 motion", 2)
    add_table(doc, ["sequence", "detector", "tracker", "HOTA", "MOTA", "IDF1", "Recall", "M1 motion", "failed"], [
        [row["sequence"], row["detector"], row["tracker"],
         fmt(row["HOTA"]), fmt(row["MOTA"]), fmt(row["IDF1"]), fmt(row["Recall"]),
         fmt(row["assertion_pass_rate"]), row["assertions_failed"]]
        for row in per_seq_m1
    ])
    heading(doc, "Detector precision / recall at IoU 0.5 (before tracking)", 2)
    add_table(doc, ["sequence", "detector", "gt", "tp", "detections", "recall", "precision"], [
        [row["sequence"], row["detector"], row["gt"], row["tp"], row["detections"],
         fmt(row["recall"]), fmt(row["precision"])]
        for row in det_recall
    ])
    heading(doc, "Detection recall by pedestrian height (IoU 0.5, all 7 videos pooled)", 2)
    pooled = defaultdict(lambda: {"gt": 0, "tp": 0})
    for row in det_size:
        key = (row["detector"], row["height_bin"])
        pooled[key]["gt"] += int(float(row["gt"]))
        pooled[key]["tp"] += int(float(row["tp"]))
    order = ("<40px", "40-80px", "80-160px", ">=160px")
    height_rows = []
    for detector in ("yolo26n_imgsz640", "yolo26n_imgsz960", "yolo26n_imgsz1280", "fasterrcnn_mv3_320"):
        for bin_name in order:
            stats = pooled.get((detector, bin_name), {"gt": 0, "tp": 0})
            rec = 100.0 * stats["tp"] / stats["gt"] if stats["gt"] else None
            height_rows.append([detector, bin_name, stats["gt"], stats["tp"], fmt(rec)])
    add_table(doc, ["detector", "height", "gt", "tp", "recall"], height_rows)
    para(doc, "Small-person recall is the YOLO-640 failure mode: pooled <40 px recall 12.9% at 640 vs 47.4% at 1280; "
         "40–80 px 39.9% vs 61.3%. Faster R-CNN 320-px input is 2.1% / 7.4% on those bins.")

    heading(doc, "7. T2 MinIO registry variants")
    para(doc, "Each scalable stage has at least four variants in MinIO bucket mot17-models "
         "(S3 API localhost:19000, console :19001) and under mot_pipeline .../registry/. "
         "A serving image loads MODEL_NAME / MODEL_VERSION — no rebuild. "
         "T2 offline accuracy was TrackEval on MOT17-02-DPM frames 1–100, recorded before the Guide v4 harness. "
         "That T2 slice is not the guide test slice; use section 3 for comparable accuracy. "
         "INT8 is ORT dynamic quantization of MatMul/Gemm; full ConvInteger is not implemented on this Mac/CoreML path.")
    det_rows = []
    for model in catalog["models"]:
        if model["stage"] != "detection":
            continue
        acc = model.get("offline_accuracy") or {}
        det_rows.append([
            model["name"], model.get("runtime", ""), model.get("imgsz", ""), model.get("kind", ""),
            model.get("size_mb", ""), fmt(acc.get("HOTA")), fmt(acc.get("MOTA")),
            fmt(acc.get("Recall")), fmt(acc.get("Precision")), fmt(model.get("test_slice_fps")),
            acc.get("test_slice", ""), model.get("source", ""),
        ])
    add_table(doc, ["name", "runtime", "imgsz", "kind", "MB", "HOTA", "MOTA", "Recall", "Prec", "FPS",
                    "T2 slice", "source"], det_rows)
    trk_rows = []
    for model in catalog["models"]:
        if model["stage"] != "tracking":
            continue
        acc = model.get("offline_accuracy") or {}
        trk_rows.append([
            model["name"], model.get("tracker", ""), model.get("size_bytes", ""),
            fmt(acc.get("HOTA")), fmt(acc.get("IDF1")), acc.get("IDSW", ""),
            acc.get("test_slice", ""), model.get("source", ""),
        ])
    add_table(doc, ["name", "tracker", "bytes", "HOTA", "IDF1", "IDSW", "T2 slice", "source"], trk_rows)

    heading(doc, "8. T3 gRPC stage servers and sticky tracker routing")
    para(doc, "Detect, track, and evaluate each run as their own process and container. "
         "They share image mot17-serve:local; the Deployment command and MODEL_NAME select the stage and registry variant. "
         "gRPC: Detector.Detect, Tracker.Track, Evaluator.Evaluate (services/proto/mot.proto). "
         "RequestBatcher waits until BATCH_SIZE or BATCH_TIMEOUT. "
         "Prometheus /metrics on :8080: mot_requests_total, mot_request_latency_seconds, mot_queue_length, "
         "mot_batch_size, mot_assertion_total{result=pass|fail}.")
    para(doc, "Video routing. Detection is stateless. The tracker is stateful: one in-memory ByteTrack/OC-SORT per stream_id. "
         "Frames of one MOT17 sequence must hit the same tracker pod, in frame_id order. "
         "The gateway hashes md5(stream_id) % TRACKER_REPLICAS onto a StatefulSet DNS name: "
         "track-{shard}.track.mot17.svc.cluster.local:50051. "
         "A live run printed MOT17-02-DPM → track-0.track.mot17.svc.cluster.local:50051 (shard 0/1). "
         "Inside a batched Track RPC the handler re-sorts by (stream_id, frame_id). "
         "Scale the tracker between videos, not mid-stream.")
    add_table(doc, ["stage", "workload", "default variant", "batch", "state", "CPU req/lim", "mem req/lim"], [
        ["detect", "Deployment", "yolo26n-pt-imgsz640", "4 / 50 ms", "stateless", "500m / 2", "1Gi / 3Gi"],
        ["track", "StatefulSet", "bytetrack", "8 / 20 ms", "per stream_id", "200m / 1", "512Mi / 1Gi"],
        ["evaluate", "Deployment", "GT + A1–A3", "1", "stateless", "200m / 1", "512Mi / 1Gi"],
        ["gateway", "Deployment", "run_pipeline.py", "n/a", "sticky hash by stream_id", "100m / 500m", "256Mi / 512Mi"],
    ])

    heading(doc, "9. T4 Kubernetes, Prometheus, Grafana, live run")
    para(doc, live.get("k8s", ""))
    para(doc, live.get("scale", ""))
    para(doc, live.get("summary", ""))
    para(doc, live.get("notes", ""))
    add_table(doc, ["workload", "cpu request", "cpu limit", "memory request", "memory limit"], [
        ["detect", "500m", "2", "1Gi", "3Gi"],
        ["track", "200m", "1", "512Mi", "1Gi"],
        ["evaluate", "200m", "1", "512Mi", "1Gi"],
        ["gateway", "100m", "500m", "256Mi", "512Mi"],
        ["prometheus", "100m", "500m", "256Mi", "512Mi"],
        ["grafana", "100m", "500m", "128Mi", "256Mi"],
    ])
    heading(doc, "Live gRPC metrics (MOT17-02, 20 then 30 frames, k8s)", 2)
    ev = live.get("evaluate") or {}
    para(doc, f"Evaluate on 30 frames: HOTA {ev.get('hota')}  MOTA {ev.get('mota')}  Recall {ev.get('recall')}  "
         f"motion pass {ev.get('assertion_pass_rate')}%  {ev.get('elapsed_s')}s ({ev.get('fps')} FPS).")
    add_table(doc, ["stage", "requests", "p50 ms", "p95 ms", "p99 ms", "queue", "batch", "CPU cores",
                    "assert pass", "assert fail", "pass rate"], [
        [stage, stats.get("requests", ""), stats.get("p50_ms", ""), stats.get("p95_ms", ""),
         stats.get("p99_ms", ""), stats.get("queue_length", ""), stats.get("batch_size", ""),
         stats.get("cpu_cores", ""), stats.get("assertion_pass", ""), stats.get("assertion_fail", ""),
         stats.get("pass_rate", "")]
        for stage, stats in live.get("stages", {}).items()
    ])
    para(doc, "Grafana dashboard MOT17 pipeline (uid mot17-pipeline) shows per stage: p50/p95/p99 latency, "
         "throughput, queue length, batch size, CPU (process_cpu_seconds_total), assertion pass rate. "
         "kubectl -n mot17 port-forward svc/grafana 3000:3000  (admin/admin, anonymous Viewer). "
         "Deploy: ./scripts/deploy.sh   Scale: ./scripts/scale_stage.sh detect 2")

    heading(doc, "10. T5 Open-loop load generator and SLA")
    para(doc, "The load generator is open-loop: it schedules each request at t0 + i / rate and does not wait "
         "for earlier replies (loadgen/open_loop.py). A closed-loop client would hide queueing delay. "
         "Each request is one frame of one MOT17 test-slice stream from eval_slices/MOT17_test "
         "(sequences 02/04/09-FRCNN second halves). Exhausted pool cycles with a new stream_id so the "
         "tracker state resets. Accuracy under load uses the same labeled boxes as the Guide v4 harness.")
    para(doc, "Calibration follows Guide v4: pick a reference (YOLO26n imgsz=640 + ByteTrack, batch 1, "
         "1 replica × 1 CPU core). Measure L0 = p95 latency at very low load. SLO = 2 × L0. "
         "T0 = highest open-loop rate with p95 ≤ SLO. Target rate = 4 × T0, which a single 1-core replica "
         "cannot sustain. The SLA unit is milliseconds per frame (p95, end to end) plus frames/s. "
         "CPU YOLO cannot hold native 30 FPS; the SLA is not a 30 FPS real-time objective.")
    if sla:
        add_table(doc, ["item", "value"], [
            ["Reference", f"{sla['reference']['detect']} + {sla['reference']['track']}, "
             f"batch {sla['reference']['batch']}, {sla['reference']['replicas']}×{sla['reference']['cores']} core"],
            ["Hardware", sla.get("hardware", "")],
            ["Load pool", sla.get("load_pool", "")],
            ["L0 p95 (low load)", f"{sla['L0_p95_ms']:.1f} ms"],
            ["SLO (2 × L0)", f"{sla['slo_p95_ms']:.1f} ms"],
            ["T0 (max rate meeting SLO, 1 replica)", f"{sla['T0_rps']:.3f} frames/s"],
            ["SLA target rate (4 × T0)", f"{sla['target_rps']:.3f} frames/s"],
            ["Frames used to calibrate", str(sla.get("n_frames", ""))],
        ])
        para(doc, sla.get("target_note", ""))
        low = sla.get("low_load") or {}
        t0s = sla.get("T0_summary") or {}
        heading(doc, "Latency breakdown at L0 and at T0", 2)
        add_table(doc, ["probe", "offered rps", "p50 ms", "p95 ms", "p99 ms",
                        "queue p95", "infer p95", "serdes p95"], [
            ["L0 low load", f"{low.get('achieved_rps', 0):.2f}",
             fmt(low.get("p50_ms"), 1), fmt(low.get("p95_ms"), 1), fmt(low.get("p99_ms"), 1),
             fmt(low.get("queue_p95_ms"), 1), fmt(low.get("infer_p95_ms"), 1),
             fmt(low.get("serdes_p95_ms"), 1)],
            ["T0 (max rate with p95 ≤ SLO)", f"{t0s.get('target_rps', sla['T0_rps']):.2f}",
             fmt(t0s.get("p50_ms"), 1), fmt(t0s.get("p95_ms"), 1), fmt(t0s.get("p99_ms"), 1),
             fmt(t0s.get("queue_p95_ms"), 1), fmt(t0s.get("infer_p95_ms"), 1),
             fmt(t0s.get("serdes_p95_ms"), 1)],
        ])
        para(doc, "At low load the p95 is almost all inference (~54 ms) plus decode (~11 ms); queue is <1 ms. "
             "At T0, queue p95 rises to ~88 ms while infer stays ~27 ms — that queueing is what a closed-loop "
             "client would hide. SLO = 127.4 ms still holds at 38.4 frames/s on one replica.")
    else:
        para(doc, "docs/sla.json is missing. Run make profile.")

    heading(doc, "11. T6 Per-stage profiling")
    para(doc, "Each scalable stage was swept on this machine's CPUs: 4 variants × 4 resource configs "
         "(1×1, 1×2, 2×1, 2×2 cores) × batch {1, 4, 8}. Cost = replicas × CPU cores per replica. "
         "Cores/replica is an explicit torch/OMP thread cap. GPU numbers are not substituted. "
         "Accuracy is looked up from the Guide v4 test slice (or T2 100-frame ONNX slice) and does not "
         "change with replica count. Tracker profiling caches YOLO26n-640 detections so the detector is "
         "not recomputed for every tracker variant. Latency is split into queueing, inference, and "
         "decode/serialization. “Max throughput meeting SLO” is the highest open-loop rate whose p95 "
         "stays within the SLA latency objective.")
    if t6:
        detect_rows = [r for r in t6 if r["stage"] == "detect"]
        track_rows = [r for r in t6 if r["stage"] == "track"]
        t6_headers = ["Stage", "Variant", "Batch", "Replicas", "Cores/replica", "Cost",
                      "p50", "p95", "p99", "Max throughput meeting SLO", "Accuracy", "Notes"]
        def t6_row(r):
            note = r.get("notes") or ""
            split = (f"queue {fmt(r['queue_p95_ms'], 0)} / infer {fmt(r['infer_p95_ms'], 0)} / "
                     f"serdes {fmt(r.get('serdes_p95_ms'), 0)} ms. {note}")
            return [
                r["stage"], r["variant"], r["batch"], r["replicas"], r["cores_per_replica"], r["cost"],
                fmt(r["p50_ms"], 0), fmt(r["p95_ms"], 0), fmt(r["p99_ms"], 0),
                fmt(r["max_throughput_slo_rps"], 2), fmt(r["accuracy_hota"], 1), split[:120],
            ]
        heading(doc, "Detect stage (assignment T6 table)", 2)
        add_table(doc, t6_headers, [t6_row(r) for r in detect_rows])
        heading(doc, "Track stage (upstream detections cached)", 2)
        add_table(doc, t6_headers, [t6_row(r) for r in track_rows])
        def mean(rows, key):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
            return sum(vals) / len(vals) if vals else 0.0
        d_infer = mean(detect_rows, "infer_p95_ms")
        t_infer = mean(track_rows, "infer_p95_ms")
        d_q = mean(detect_rows, "queue_p95_ms")
        para(doc, f"Bottleneck: the detector. Mean detect infer p95 ≈ {d_infer:.0f} ms vs tracker "
             f"infer p95 ≈ {t_infer:.0f} ms. Detect queue p95 ≈ {d_q:.0f} ms once the open-loop rate "
             f"exceeds one replica's service rate. Network/decode (serdes) is small next to YOLO. "
             f"Adding a second detect replica raises SLO throughput roughly in proportion to replicas; "
             f"a second core on the same replica helps less because YOLO26n on CPU is not fully "
             f"linear in threads. Batch size {1,4,8} barely changes CPU YOLO throughput: the handler "
             f"still runs images sequentially inside the batch (no fused GPU batch). Larger batches "
             f"only hide the batcher timeout under bursty arrivals.")
        para(doc, "Accuracy is constant across replica counts for a given variant (HOTA column repeats). "
             "If it moved with replicas the measurement would be wrong.")
        plots = ROOT / "docs" / "plots"
        add_picture(doc, plots / "t6_accuracy_vs_p95.png",
                    "Figure. T6 accuracy vs p95 latency. Horizontal bands mean HOTA does not change with replicas.")
        add_picture(doc, plots / "t6_accuracy_vs_throughput.png",
                    "Figure. T6 accuracy vs max open-loop throughput that still meets the SLO.")
        add_picture(doc, plots / "t6_variant_accuracy_vs_p95.png",
                    "Figure. Per-variant accuracy vs p95 at 1 replica × 1 core, batch 4. imgsz=1280 is the accuracy lever.")
    else:
        para(doc, "docs/profile/t6_stage_profile.csv is missing. Run make profile.")

    heading(doc, "12. T7 End-to-end configurations, gold-input ablation, Pareto")
    para(doc, "End-to-end cost is detect replicas×cores plus track replicas×cores. "
         "The 4×4 detect×track cross-product is scored for accuracy from the Guide v4 test slice "
         "(ONNX HOTA from the T2 100-frame slice). Live detect+track was run at the SLA target rate "
         "for every detector × {ByteTrack, OC-SORT} on two topologies (1×1+1×1 cost 2, and 2×1+1×1 cost 3). "
         "Gold-input ablation feeds GT pedestrian boxes to the tracker and skips the detector.")
    if t7:
        live = [r for r in t7 if r.get("source") == "live_detect"]
        gold = [r for r in t7 if r.get("source") == "gold_input"]
        cached = [r for r in t7 if r.get("source") == "cached_detect"]
        heading(doc, "Gold-input ablation", 2)
        add_table(doc, ["detect", "track", "cost", "p95 ms", "HOTA", "meets SLA"], [
            [r["detect"], r["track"], r["cost"], fmt(r["p95_ms"], 0), fmt(r["hota"], 1), r["meets_sla"]]
            for r in gold
        ])
        if gold and live:
            gold_h = float(gold[0]["hota"])
            best_pipe = max(float(r["hota"]) for r in live)
            para(doc, f"Gold-input HOTA {gold_h:.1f} vs best live pipeline HOTA {best_pipe:.1f}. "
                 f"Almost all identity/detection error is the detector: feeding GT boxes recovers ~97 HOTA, "
                 f"so swapping trackers on YOLO output cannot close that gap.")
        heading(doc, "Accuracy of every detect × track pair (cached upstream)", 2)
        add_table(doc, ["detect", "track", "HOTA", "note"], [
            [r["detect"], r["track"], fmt(r["hota"], 1), (r.get("accuracy_note") or "")[:80]]
            for r in cached
        ])
        heading(doc, "Live end-to-end at the SLA target rate", 2)
        add_table(doc, ["detect", "track", "det n×c", "trk n×c", "cost", "p50", "p95", "p99",
                        "HOTA", "meets SLA"], [
            [r["detect"], r["track"],
             f"{r.get('detect_replicas', r['replicas'])}×{r.get('detect_cores', r['cores_per_replica'])}",
             f"{r.get('track_replicas', 1)}×{r.get('track_cores', 1)}",
             r["cost"], fmt(r["p50_ms"], 0), fmt(r["p95_ms"], 0), fmt(r["p99_ms"], 0),
             fmt(r["hota"], 1), r["meets_sla"]]
            for r in live
        ])
        meeting = [r for r in live if str(r["meets_sla"]).lower() == "true"]
        para(doc, f"{len(meeting)} of {len(live)} live detect+track configurations meet the SLA at the "
             f"target rate of {fmt(sla['target_rps'] if sla else live[0]['target_rps'], 1)} frames/s. "
             "That is expected: the target is 4×T0, and T6 shows one YOLO-640 replica sustains about 34 frames/s "
             "inside the SLO, so about five detect replicas are required. Two and even four replicas still "
             "queue (p95 seconds, not milliseconds) because offered load 153 frames/s exceeds 2×34 and 4×34. "
             "Meeting the SLA is a property of detect replica count and detector service time, not of the "
             "tracker: ByteTrack vs OC-SORT changes HOTA by about one point. Accuracy is identical at cost 2, 3 and 5 "
             "for a given variant pair — if HOTA moved with replicas the measurement would be wrong.")
        # Does upstream change best downstream?
        by_det = {}
        for r in cached:
            by_det.setdefault(r["detect"], []).append(r)
        lines = []
        for det, group in by_det.items():
            best = max(group, key=lambda x: float(x["hota"]))
            lines.append(f"{det}: best tracker {best['track']} (HOTA {float(best['hota']):.1f})")
        para(doc, "Does the upstream choice change the best downstream choice? " + " ".join(lines) + ". "
             "On the Guide test slice, imgsz=1280 prefers OC-SORT (47.0 vs ByteTrack 46.7) while imgsz=640 "
             "prefers ByteTrack (36.6 vs OC-SORT 34.7). The ranking is not identical across detectors, "
             "but the HOTA gap is small next to the detector gap (640 → 1280 is +10 HOTA).")
        para(doc, "What is “optimal”? It depends on the load. At the SLA target rate, any 1-replica detect "
             "topology is over capacity (p95 blows up from queueing). At a load below T0, the cheapest "
             "1×1 YOLO-1280 + OC-SORT is both most accurate among YOLO configs and latency-feasible. "
             "Public FRCNN + ByteTrack (HOTA 50.1) is more accurate still but is not a served detector "
             "variant in this stack. There is no single optimum independent of offered load.")
        add_picture(doc, ROOT / "docs" / "plots" / "t7_cost_accuracy_pareto.png",
                    "Figure. T7 end-to-end cost vs HOTA. Green triangle = gold-input ablation (not deployable). "
                    "Red diamonds = live detect+track at the SLA target rate (none meet 153.6 frames/s with ≤4 detect replicas). "
                    "Dashed line = live Pareto front (YOLO-1280 + OC-SORT at cost 2).")
    else:
        para(doc, "docs/profile/t7_e2e.csv is missing. Run make profile.")

    heading(doc, "13. Deploy recommendation")
    rec = None
    if t7:
        live = [r for r in t7 if r.get("source") == "live_detect"]
        meeting = [r for r in live if str(r["meets_sla"]).lower() == "true"]
        pool = meeting or live
        if pool:
            rec = max(pool, key=lambda r: (float(r["hota"]), -float(r["cost"])))
    if rec and sla:
        para(doc, f"SLA: p95 ≤ {sla['slo_p95_ms']:.0f} ms at {sla['target_rps']:.1f} frames/s "
             f"(L0 {sla['L0_p95_ms']:.0f} ms, T0 {sla['T0_rps']:.1f} frames/s on 1×1 YOLO-640 + ByteTrack).")
        para(doc, "No 1–4 replica live config meets that target rate on this CPU. T6 max SLO throughput for "
             "YOLO-640 batch-1 is ~34 frames/s per replica (~58 at 2×1, ~64 at 2×2). "
             "ceil(153.6 / 34.3) = 5 detect replicas of YOLO-640. YOLO-1280 is ~7.8 frames/s per replica, "
             "so it would need ~20 replicas to hit the same rate — not a latency-feasible accuracy upgrade "
             "at this SLA.")
        para(doc, "Deploy at the SLA target: detect yolo26n-pt-imgsz640, 5 replicas × 1 CPU core, batch 1; "
             "track ByteTrack, 1 replica × 1 core; cost 6 replica-cores; Guide-slice HOTA 36.6. "
             "Scale with ./scripts/scale_stage.sh detect 5. Do not scale the tracker mid-stream.")
        para(doc, "If the offered load drops to T0 (~38 frames/s), the accuracy-optimal deployable config is "
             "YOLO-1280 + OC-SORT at 1 detect replica (cost 2, HOTA 47.0), which T6 shows can meet the "
             "p95 SLO at ~8 frames/s and needs 5 replicas to approach T0. Public FRCNN + ByteTrack (HOTA 50.1) "
             "is more accurate still but is not a served detector in this stack. "
             "Gold-input HOTA 97.3 is the tracker/association ceiling: almost all remaining error is the detector.")
    else:
        para(doc, "Run make profile to fill the recommendation from measured T7 rows.")

    heading(doc, "14. How to change a variant without rebuilding")
    para(doc, "1. Confirm the name in registry/catalog.json (or MinIO mot17-models).")
    para(doc, "2. kubectl -n mot17 set env deployment/detect MODEL_NAME=yolo26n-pt-imgsz1280")
    para(doc, "3. The detect pod restarts, pulls that artifact from MinIO, and serves it. Image tag mot17-serve:local is unchanged.")
    para(doc, "Tracker: kubectl -n mot17 set env statefulset/track MODEL_NAME=ocsort")

    heading(doc, "15. Presentation outline")
    para(doc, "1. Pipeline: MOT17 person detect → track → TrackEval + A1–A3. CPU-only, pretrained, no train.")
    para(doc, "2. Harness: Guide v4 02/04/09 FRCNN halves; HOTA combined; baseline public FRCNN 50.1; gold 97.3.")
    para(doc, "3. Assertions anti-correlate with HOTA (Spearman −0.89) because A2 punishes high-recall squat boxes.")
    para(doc, "4. Variants in MinIO; gRPC + batching; sticky md5(stream_id) tracker routing; k8s + Prom/Grafana.")
    para(doc, "5. Open-loop load on the labeled test slice; SLA = 2×L0 p95 at 4×T0, which needs >1 detect replica.")
    para(doc, "6. Profiling: detector is the bottleneck (infer + queue); tracker is cheap; accuracy is variant-not-replica.")
    para(doc, "7. Pareto / deploy: pick the highest-HOTA live config that meets the SLA, usually YOLO-1280 + OC-SORT with scaled detect.")

    heading(doc, "Source files")
    add_table(doc, ["path", "role"], [
        ["eval_slices/manifest.json", "Guide v4 frame IDs + MOT17 file hashes"],
        ["eval_slices/DATASET_VERSION.txt", "Dataset version / split rule"],
        ["mot_pipeline .../slices.py, build_slices.py, eval_harness.py, evaluate.py", "harness"],
        ["mot_pipeline .../results/guide_eval/", "test-slice CSV/JSON"],
        ["mot_pipeline .../results/summary.csv, per_sequence.csv", "T1 seven-sequence TrackEval"],
        ["mot_pipeline .../registry/catalog.json", "T2 variants"],
        ["services/proto/mot.proto, services/*/server.py, services/*/Dockerfile", "T3 gRPC"],
        ["k8s/pipeline.yaml, k8s/monitoring.yaml, k8s/grafana-dashboard.json", "T4"],
        ["scripts/scale_stage.sh", "scripted kubectl scale"],
        ["loadgen/open_loop.py, loadgen/pool.py", "T5 open-loop + labeled pool"],
        ["profiling/*.py, run_profile.sh, Makefile", "T5–T7 entry point"],
        ["docs/sla.json, docs/profile/, docs/plots/", "regenerated tables and figures"],
        ["docs/live_metrics.json", "k8s Prometheus scrape of the live run"],
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
