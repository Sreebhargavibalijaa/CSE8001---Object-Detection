"""Load a named registry variant without rebuilding the image."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from registry import ModelRegistry  # noqa: E402

CACHE = Path(os.environ.get("MODEL_CACHE", "/tmp/mot-models"))


def load_variant(stage, name=None, version=None):
    name = name or os.environ.get("MODEL_NAME")
    version = version or os.environ.get("MODEL_VERSION", "1.0.0")
    if not name:
        raise SystemExit(f"{stage} service requires MODEL_NAME")
    registry = ModelRegistry()
    meta = registry.get_metadata(stage, name, version)
    dest = CACHE / stage / name / version / meta["artifact"]
    registry.get_file(stage, name, version, meta["artifact"], dest)
    print(f"loaded {stage}/{name}/{version} -> {dest} ({dest.stat().st_size} bytes)", flush=True)
    return dest, meta
