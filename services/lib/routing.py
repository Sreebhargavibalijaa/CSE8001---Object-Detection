"""Sticky routing for the stateful tracker.

Every frame carries stream_id (the MOT17 sequence name). The gateway hashes
that id onto a tracker StatefulSet ordinal so one stream always hits the same
pod, in order:

    shard = md5(stream_id) % TRACKER_REPLICAS
    track-{shard}.track.<namespace>.svc.cluster.local:50051

Detection is stateless and can land on any detect replica.
Scaling the tracker mid-stream remaps hashes; do it between videos.
"""

from __future__ import annotations

import hashlib
import os


def tracker_shard(stream_id, replicas=None):
    replicas = int(os.environ.get("TRACKER_REPLICAS", replicas or 1))
    replicas = max(1, replicas)
    digest = hashlib.md5(stream_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % replicas, replicas


def tracker_target(stream_id, replicas=None, service=None, namespace=None, port=None):
    shard, replicas = tracker_shard(stream_id, replicas)
    service = service or os.environ.get("TRACKER_SERVICE", "track")
    namespace = namespace or os.environ.get("TRACKER_NAMESPACE", "mot17")
    port = int(os.environ.get("TRACKER_PORT", port or 50051))
    host = os.environ.get("TRACKER_ADDR")
    if host:
        return host, shard, replicas
    if replicas == 1 and os.environ.get("TRACKER_SINGLE", ""):
        return os.environ["TRACKER_SINGLE"], shard, replicas
    return f"{service}-{shard}.{service}.{namespace}.svc.cluster.local:{port}", shard, replicas
