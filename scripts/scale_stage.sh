#!/usr/bin/env bash
# Scripted scale for a pipeline stage (T4). Tracker uses a StatefulSet so
# stream-id hashing stays stable; detect/evaluate are Deployments.
set -euo pipefail

NS="${NS:-mot17}"
STAGE="${1:?usage: scale_stage.sh <detect|track|evaluate> <replicas>}"
REPLICAS="${2:?usage: scale_stage.sh <detect|track|evaluate> <replicas>}"

case "$STAGE" in
  detect|evaluate)
    kubectl -n "$NS" scale deployment "$STAGE" --replicas="$REPLICAS"
    kubectl -n "$NS" rollout status deployment "$STAGE"
    ;;
  track)
    kubectl -n "$NS" scale statefulset track --replicas="$REPLICAS"
    kubectl -n "$NS" rollout status statefulset track
    kubectl -n "$NS" set env deployment/gateway TRACKER_REPLICAS="$REPLICAS"
    kubectl -n "$NS" rollout status deployment gateway
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac

echo "scaled $STAGE -> $REPLICAS"
kubectl -n "$NS" get deploy,sts
