#!/usr/bin/env bash
# Build the serving image, import it into k3s and deploy pipeline + Prometheus + Grafana.
# Thin wrapper around the matching steps of run_profile_k8s.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./run_profile_k8s.sh image deploy
echo
echo "Grafana:    kubectl -n mot17 port-forward svc/grafana 3000:3000   (sign in admin / admin)"
echo "Prometheus: kubectl -n mot17 port-forward svc/prometheus 9090:9090"
echo "Scale:      ./scripts/scale_stage.sh detect 2"
