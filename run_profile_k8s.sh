#!/usr/bin/env bash
# Single entry point on the course VM (k3s): rebuilds every Milestone 2 T5-T7 table
# and plot from the deployed Kubernetes pipeline. Nothing is changed by hand: every
# variant / batch / replica / core setting is applied with kubectl from
# profiling/k8s_profile.py. Safe to re-run: finished T6/T7 points are skipped.
#
#   ./run_profile_k8s.sh                 # all steps, in order
#   ./run_profile_k8s.sh t6 t7 plots     # only these steps
#
# Steps:
#   slices     build Guide v4 dev/test slices (eval_slices/) if missing
#   registry   start MinIO, export ONNX / static INT8 / Faster R-CNN, register all variants
#   offline    test-slice accuracy for every detector x tracker + baseline + gold input
#   image      build the serving image and import it into k3s
#   deploy     apply k8s manifests (pipeline + Prometheus + Grafana) and restart the stages
#   preflight  ~10 min: every variant loads in a 1-core pod, answers, and is not CPU-throttled
#   calibrate  T5: L0, SLO = 2 x L0, T0, target = 4 x T0  -> docs/sla.json
#   t6         per-stage sweep                            -> docs/profile/t6_stage_profile.csv
#   t7         end-to-end configs at 1 x T0 and 4 x T0    -> docs/profile/t7_e2e*.csv
#   plots      the three required plots                   -> docs/plots/
#   restore    put the deployment back to the manifest defaults
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PIPE="$ROOT/mot_pipeline (Object_Detection + Tracking)"
PY="${PYTHON:-$PIPE/venv/bin/python3}"
export PYTHONPATH="$ROOT:$PIPE"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/docs/.mpl}"
mkdir -p docs/profile docs/plots "$MPLCONFIGDIR"

LOG="docs/profile/run_k8s_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

STEPS=("$@")
if [[ ${#STEPS[@]} -eq 0 ]]; then
  STEPS=(slices registry offline image deploy preflight calibrate t6 t7 plots restore)
fi
want() { local s; for s in "${STEPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
banner() { echo; echo "==== $(date '+%F %T')  $*"; }
NS=mot17

banner "steps: ${STEPS[*]}   log: $LOG"
FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
echo "free disk: ${FREE_GB} GB"
if (( FREE_GB < 4 )); then
  echo "less than 4 GB free on /; free space first (docker image prune -f)"; exit 1
fi

if want slices; then
  banner "slices"
  if [[ -d eval_slices/MOT17_test/train && -d eval_slices/MOT17_development/train ]]; then
    echo "eval_slices already built"
  else
    (cd "$PIPE" && "$PY" build_slices.py)
  fi
fi

if want registry; then
  banner "registry (MinIO + variants)"
  docker compose up -d minio
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:19000/minio/health/live >/dev/null && break
    sleep 2
  done
  (cd "$PIPE" && "$PY" register_variants.py --device cpu)
fi

if want offline; then
  banner "offline accuracy on the Guide v4 test slice"
  (cd "$PIPE" && "$PY" guide_offline.py)
fi

if want image; then
  banner "serving image -> k3s"
  docker build -f services/Dockerfile -t mot17-serve:local .
  docker save mot17-serve:local | sudo k3s ctr images import -
fi

if want deploy; then
  banner "deploy"
  kubectl apply -f k8s/pipeline.yaml
  kubectl apply -f k8s/monitoring.yaml
  kubectl -n "$NS" rollout restart deployment/detect deployment/evaluate deployment/gateway statefulset/track
  for w in deployment/detect statefulset/track deployment/evaluate deployment/gateway; do
    kubectl -n "$NS" rollout status "$w" --timeout=600s
  done
  kubectl -n "$NS" get pods
fi

if want preflight; then
  banner "preflight (every variant in a 1-core pod)"
  "$PY" -m profiling.k8s_profile preflight
fi

if want calibrate; then
  banner "T5 SLA calibration"
  "$PY" -m profiling.k8s_profile calibrate
fi

if want t6; then
  banner "T6 per-stage sweep"
  "$PY" -m profiling.k8s_profile stages --resume
fi

if want t7; then
  banner "T7 end-to-end"
  "$PY" -m profiling.k8s_profile e2e --resume
fi

if want plots; then
  banner "plots"
  "$PY" -m profiling.k8s_plots --profile docs/profile --out docs/plots \
    || echo "PLOTS FAILED (results are safe; re-run: ./run_profile_k8s.sh plots)"
fi

if want restore; then
  banner "restore manifest defaults"
  # The sweep patched env / resources / replicas; recreate the two stages from the manifest.
  kubectl -n "$NS" delete deployment/detect statefulset/track --wait=true
  kubectl apply -f k8s/pipeline.yaml
  kubectl -n "$NS" rollout status deployment/detect --timeout=600s
  kubectl -n "$NS" rollout status statefulset/track --timeout=600s
fi

banner "done"
echo "  SLA     docs/sla.json"
echo "  T6      docs/profile/t6_stage_profile.csv"
echo "  T7      docs/profile/t7_e2e.csv  (+ t7_e2e_all_loads.csv, t7_summary.json)"
echo "  plots   docs/plots/"
echo "  offline mot_pipeline (Object_Detection + Tracking)/results/guide_eval_vm/"
echo "  log     $LOG"
