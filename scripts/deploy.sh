#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> building mot17-serve:local"
docker build -f services/Dockerfile -t mot17-serve:local .

echo "==> applying kubernetes manifests"
kubectl apply -f k8s/pipeline.yaml
kubectl apply -f k8s/monitoring.yaml

echo "==> waiting for stage pods"
kubectl -n mot17 rollout status deployment/detect --timeout=180s
kubectl -n mot17 rollout status statefulset/track --timeout=180s
kubectl -n mot17 rollout status deployment/evaluate --timeout=180s
kubectl -n mot17 rollout status deployment/prometheus --timeout=180s
kubectl -n mot17 rollout status deployment/grafana --timeout=180s

echo "==> pods"
kubectl -n mot17 get pods,svc
echo
echo "Grafana: kubectl -n mot17 port-forward svc/grafana 3000:3000"
echo "Prometheus: kubectl -n mot17 port-forward svc/prometheus 9090:9090"
echo "Run a sequence:"
echo "  kubectl -n mot17 exec deploy/gateway -- python -m services.gateway.run_pipeline --sequence MOT17-02-DPM --max-frames 30"
echo "Scale a stage:"
echo "  ./scripts/scale_stage.sh detect 2"
