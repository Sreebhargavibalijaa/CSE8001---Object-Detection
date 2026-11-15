"""Shared gRPC + metrics process bootstrap."""

from __future__ import annotations

import os
from concurrent import futures

import grpc

from services.lib.metrics import serve_metrics


def serve(add_servicer, metrics_port=None, grpc_port=None, workers=8):
    metrics_port = int(os.environ.get("METRICS_PORT", metrics_port or 8080))
    grpc_port = int(os.environ.get("GRPC_PORT", grpc_port or 50051))
    serve_metrics(metrics_port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(os.environ.get("GRPC_WORKERS", workers))))
    add_servicer(server)
    server.add_insecure_port(f"0.0.0.0:{grpc_port}")
    print(f"gRPC :{grpc_port}  metrics :{metrics_port}", flush=True)
    server.start()
    server.wait_for_termination()
