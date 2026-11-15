"""Regenerate SLA, T6, T7, plots. Invoked by run_profile.sh / make profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(module, extra):
    cmd = [sys.executable, "-m", module, *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    run("profiling.calibrate", [
        "--frames", str(args.frames), "--replays", str(args.replays), "--out", "docs/sla.json",
    ])
    run("profiling.profile_stages", [
        "--frames", str(args.frames), "--replays", str(args.replays),
        "--rounds", str(args.rounds), "--out", "docs/profile", "--sla", "docs/sla.json",
    ])
    run("profiling.profile_e2e", [
        "--frames", str(args.frames), "--replays", str(args.replays),
        "--out", "docs/profile", "--sla", "docs/sla.json",
    ])
    run("profiling.plots", ["--profile", "docs/profile", "--out", "docs/plots"])
    report = ROOT / "scripts" / "write_report.py"
    subprocess.check_call([sys.executable, str(report)], cwd=ROOT)


if __name__ == "__main__":
    main()
