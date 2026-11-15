"""T6/T7 plots: accuracy vs p95, accuracy vs throughput, cost–accuracy Pareto."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row, key):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    return float(value)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def pareto_front(points):
    """points: list of (cost, hota, row). Minimize cost, maximize hota."""
    ordered = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best_hota = [], -1e9
    for cost, hota, row in ordered:
        if hota > best_hota:
            front.append((cost, hota, row))
            best_hota = hota
    return front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=ROOT / "docs" / "profile")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "plots")
    args = parser.parse_args()
    t6 = read_csv(args.profile / "t6_stage_profile.csv")
    t7 = read_csv(args.profile / "t7_e2e.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for stage, marker in (("detect", "o"), ("track", "s")):
        rows = [r for r in t6 if r["stage"] == stage]
        ax.scatter([fnum(r, "p95_ms") for r in rows], [fnum(r, "accuracy_hota") for r in rows],
                   marker=marker, label=stage, alpha=0.75)
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("HOTA")
    ax.set_title("Accuracy vs p95 latency (all T6 points)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, args.out / "t6_accuracy_vs_p95.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for stage, marker in (("detect", "o"), ("track", "s")):
        rows = [r for r in t6 if r["stage"] == stage]
        ax.scatter([fnum(r, "max_throughput_slo_rps") for r in rows],
                   [fnum(r, "accuracy_hota") for r in rows],
                   marker=marker, label=stage, alpha=0.75)
    ax.set_xlabel("Max throughput meeting SLO (frames/s)")
    ax.set_ylabel("HOTA")
    ax.set_title("Accuracy vs max SLO throughput (all T6 points)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, args.out / "t6_accuracy_vs_throughput.png")

    # One marker per detect variant (median p95 at 1 replica, 1 core, batch 4) for a cleaner variant plot.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for stage in ("detect", "track"):
        grouped = {}
        for row in t6:
            if row["stage"] != stage:
                continue
            if int(row["replicas"]) != 1 or int(float(row["cores_per_replica"])) != 1 or int(row["batch"]) != 4:
                continue
            grouped[row["variant"]] = row
        xs = [fnum(r, "p95_ms") for r in grouped.values()]
        ys = [fnum(r, "accuracy_hota") for r in grouped.values()]
        ax.scatter(xs, ys, label=stage, s=70)
        for row in grouped.values():
            ax.annotate(row["variant"].replace("yolo26n-", "").replace("yolo26n_", ""),
                        (fnum(row, "p95_ms"), fnum(row, "accuracy_hota")), fontsize=7)
    ax.set_xlabel("p95 latency (ms) at 1 replica × 1 core, batch 4")
    ax.set_ylabel("HOTA")
    ax.set_title("Per-variant accuracy vs p95")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, args.out / "t6_variant_accuracy_vs_p95.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    live = [r for r in t7 if r.get("source") == "live_detect"]
    gold = [r for r in t7 if r.get("source") == "gold_input"]
    if gold:
        ax.scatter([fnum(r, "cost") for r in gold], [fnum(r, "hota") for r in gold],
                   c="#2ca02c", marker="^", s=90, zorder=3, label="gold-input ablation (not deployable)")
        ax.axhline(fnum(gold[0], "hota"), color="#2ca02c", ls=":", alpha=0.5)
    if live:
        colors = ["#1f77b4" if r["meets_sla"] in ("True", "true", True) else "#d62728" for r in live]
        ax.scatter([fnum(r, "cost") for r in live], [fnum(r, "hota") for r in live],
                   c=colors, marker="D", s=80, label="live detect+track (blue=meets SLA)")
        labeled = set()
        for row in live:
            key = row["detect"].replace("yolo26n_", "").replace("yolo26n-", "")[:12]
            if key in labeled:
                continue
            labeled.add(key)
            ax.annotate(key, (fnum(row, "cost"), fnum(row, "hota")), fontsize=7)
    # Pareto of deployable (live) configs only. Gold dominates if included.
    points = [(fnum(r, "cost"), fnum(r, "hota"), r) for r in live
              if fnum(r, "cost") is not None and fnum(r, "hota") is not None]
    front = pareto_front(points)
    if front:
        ax.plot([p[0] for p in front], [p[1] for p in front], "k--", label="live Pareto front")
        for cost, hota, row in front:
            ax.annotate(f"{row['detect'].split('_')[-1]}+{row['track']}", (cost, hota),
                        fontsize=7, xytext=(4, 6), textcoords="offset points")
    ax.set_xlabel("Cost (detect replica-cores + track replica-cores)")
    ax.set_ylabel("HOTA")
    ax.set_title("End-to-end cost vs accuracy (SLA at target rate)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, args.out / "t7_cost_accuracy_pareto.png")

    (args.out / "manifest.json").write_text(json.dumps({
        "t6_rows": len(t6), "t7_rows": len(t7), "pareto": [
            {"cost": c, "hota": h, "detect": r["detect"], "track": r["track"], "meets_sla": r["meets_sla"]}
            for c, h, r in front
        ],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
