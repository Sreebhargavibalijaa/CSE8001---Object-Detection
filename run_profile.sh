#!/usr/bin/env bash
# Single entry point: regenerate every T5–T7 table and plot, then the Word report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/mot_pipeline (Object_Detection + Tracking)/venv/bin/python3}"
export PYTHONPATH="$ROOT:$ROOT/mot_pipeline (Object_Detection + Tracking)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/docs/.mpl}"
mkdir -p "$MPLCONFIGDIR" docs/profile docs/plots

FRAMES="${PROFILE_FRAMES:-8}"
REPLAYS="${PROFILE_REPLAYS:-2}"
ROUNDS="${PROFILE_ROUNDS:-3}"

echo "==> slices"
if [[ ! -f eval_slices/manifest.json ]]; then
  "$PY" "mot_pipeline (Object_Detection + Tracking)/build_slices.py"
fi

echo "==> T5 SLA calibration"
"$PY" -m profiling.calibrate --frames "$FRAMES" --replays "$REPLAYS" --out docs/sla.json

echo "==> T6 stage profile"
"$PY" -m profiling.profile_stages --frames "$FRAMES" --replays "$REPLAYS" --rounds "$ROUNDS" --out docs/profile --sla docs/sla.json

echo "==> T7 end-to-end"
"$PY" -m profiling.profile_e2e --frames "$FRAMES" --replays "$REPLAYS" --out docs/profile --sla docs/sla.json

echo "==> plots"
"$PY" -m profiling.plots --profile docs/profile --out docs/plots

echo "==> Word report"
"$PY" scripts/write_report.py

echo "done"
echo "  SLA     docs/sla.json"
echo "  T6/T7   docs/profile/"
echo "  plots   docs/plots/"
echo "  report  CSE8001_Milestone2_All_Results.docx"
