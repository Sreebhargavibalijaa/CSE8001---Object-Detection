"""Milestone 2 plots from the k3s profiling CSVs.

  t6_accuracy_vs_p95.png          accuracy vs p95 latency, one point per variant
  t6_accuracy_vs_throughput.png   accuracy vs max throughput meeting the SLO, per variant
  t7_cost_accuracy_pareto.png     end-to-end cost vs accuracy, SLA marked, Pareto front

Per-variant points use the reference resources (1 replica x 1 core). Detector and
tracker panels have separate x-axes because their latencies differ by ~100x.
Rows with missing values (configs that could not be deployed) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8982"
STAGE_STYLE = {"detect": (BLUE, "o", "Detector variants"), "track": (ORANGE, "s", "Tracker variants")}


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row, key):
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def short(name):
    return name.replace("yolo26n-", "").replace("-imgsz", " ").replace("fasterrcnn-mv3-320", "frcnn-mv3 320")


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def label_points(ax, points):
    """Direct labels, alternating above/below so close points do not collide."""
    ordered = sorted(points, key=lambda p: (p[1], p[0]))
    for index, (x, y, text) in enumerate(ordered):
        dy = 8 if index % 2 == 0 else -12
        ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                    fontsize=7.5, color=INK)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def reference_rows(t6, stage):
    """Per variant: rows at 1 replica x 1 core, keyed by batch."""
    out = {}
    for row in t6:
        if row["stage"] != stage or row["replicas"] != "1" or row["cores_per_replica"] != "1":
            continue
        out.setdefault(row["variant"], {})[int(row["batch"])] = row
    return out


def per_variant_plot(t6, out, x_of, xlabel, title, note):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    drawn = False
    for ax, stage in zip(axes, ("detect", "track")):
        color, marker, name = STAGE_STYLE[stage]
        points = []
        for variant, by_batch in reference_rows(t6, stage).items():
            x, y = x_of(by_batch), None
            for row in by_batch.values():
                y = num(row, "accuracy_hota") if y is None else y
            if x is not None and y is not None:
                points.append((x, y, short(variant)))
        style(ax, xlabel, "HOTA (test slice)", name)
        if points:
            drawn = True
            ax.scatter([p[0] for p in points], [p[1] for p in points], s=70, marker=marker,
                       color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            label_points(ax, points)
            ax.margins(x=0.18, y=0.22)
        else:
            ax.text(0.5, 0.5, "no measured points", transform=ax.transAxes, ha="center",
                    color=MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out)
    return drawn


def low_load_p95(by_batch):
    row = by_batch.get(1) or next(iter(by_batch.values()))
    return num(row, "low_load_p95_ms")


def best_throughput(by_batch):
    values = [num(row, "max_throughput_slo_rps") for row in by_batch.values()]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def pareto_front(points):
    """points: (cost, hota, row). Lowest cost first; keep each strict accuracy gain."""
    front, best = [], float("-inf")
    for cost, hota, row in sorted(points, key=lambda p: (p[0], -p[1])):
        if hota > best:
            front.append((cost, hota, row))
            best = hota
    return front


def pareto_panel(ax, rows, title, references):
    style(ax, "Cost (replicas x cores, detector + tracker)", "HOTA (test slice)", title)
    live = [(num(r, "cost"), num(r, "hota"), r) for r in rows if r.get("source") == "live_detect"]
    live = [p for p in live if p[0] is not None and p[1] is not None]
    meets = [p for p in live if p[2].get("meets_sla") in ("True", "true")]
    misses = [p for p in live if p not in meets]
    if misses:
        ax.scatter([p[0] for p in misses], [p[1] for p in misses], s=46, marker="x",
                   color=GRAY, linewidth=1.5, zorder=2, label="misses the SLA")
    if meets:
        ax.scatter([p[0] for p in meets], [p[1] for p in meets], s=70, marker="o", color=BLUE,
                   edgecolor=SURFACE, linewidth=2, zorder=3, label="meets the SLA")
    front = pareto_front(meets)
    if front:
        ax.plot([p[0] for p in front], [p[1] for p in front], color=INK, linewidth=1.4,
                linestyle="--", zorder=2.5, label="Pareto front (meets SLA)")
        label_points(ax, [(c, h, f"{short(r['detect'])} + {r['track']}") for c, h, r in front])
    for name, value in references:
        if value is not None:
            ax.axhline(value, color=MUTED, linewidth=0.9, linestyle=":", zorder=1)
            ax.annotate(f"{name}: {value:.1f}", (1.0, value), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right",
                        fontsize=7.5, color=MUTED)
    if not live:
        ax.text(0.5, 0.5, "no deployed configurations", transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=9)
    else:
        ax.margins(x=0.12, y=0.18)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    return front


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=ROOT / "docs" / "profile")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "plots")
    parser.add_argument("--sla", type=Path, default=ROOT / "docs" / "sla.json")
    args = parser.parse_args()
    t6 = read_csv(args.profile / "t6_stage_profile.csv")
    t7_target = read_csv(args.profile / "t7_e2e.csv")
    t7_all = read_csv(args.profile / "t7_e2e_all_loads.csv")
    sla = json.loads(args.sla.read_text(encoding="utf-8")) if args.sla.is_file() else {}
    slo = sla.get("slo_p95_ms")
    slo_text = "" if slo is None else f"  SLO p95 = {slo:.0f} ms."

    per_variant_plot(
        t6, args.out / "t6_accuracy_vs_p95.png", low_load_p95,
        "p95 latency at low load (ms), 1 replica x 1 core, batch 1",
        "Accuracy vs p95 latency per variant",
        "Detector HOTA: tracked with ByteTrack. Tracker HOTA: on cached YOLO26n-640 detections." + slo_text)
    per_variant_plot(
        t6, args.out / "t6_accuracy_vs_throughput.png", best_throughput,
        "Max throughput meeting the SLO (frames/s), 1 replica x 1 core, best batch",
        "Accuracy vs max throughput per variant",
        "A variant at 0 frames/s cannot meet the SLO on one core even at low load." + slo_text)

    references = []
    for name, detect in (("gold input", "gold_gt"), ("baseline", "public_frcnn")):
        values = [num(r, "hota") for r in t7_target if r.get("detect") == detect]
        values = [v for v in values if v is not None]
        references.append((name, max(values) if values else None))
    loads = sorted({r["load_multiplier"] for r in t7_all if r.get("load_multiplier")}, key=float)
    panels = [(m, [r for r in t7_all if r.get("load_multiplier") == m]) for m in loads]
    if not panels:
        panels = [("", t7_target)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), squeeze=False)
    fronts = {}
    for ax, (multiplier, rows) in zip(axes[0], panels):
        rate = next((num(r, "target_rps") for r in rows if num(r, "target_rps") is not None), None)
        title = "End-to-end" if not multiplier else (
            f"Load {float(multiplier):g} x T0" + ("" if rate is None else f" = {rate:.1f} frames/s"))
        front = pareto_panel(ax, rows, title, references)
        fronts[multiplier or "target"] = [
            {"cost": c, "hota": h, "detect": r["detect"], "track": r["track"]} for c, h, r in front]
    fig.suptitle("End-to-end cost vs accuracy" + slo_text.rstrip("."), x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, args.out / "t7_cost_accuracy_pareto.png")
    (args.out / "manifest.json").write_text(json.dumps(
        {"t6_rows": len(t6), "t7_rows": len(t7_all) or len(t7_target), "pareto_fronts": fronts},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
