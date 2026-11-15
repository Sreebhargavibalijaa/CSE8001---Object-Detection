"""Labeled load pool: Guide v4 test-slice sequences replayed as streams.

Each request is one frame of one stream. A new replay uses a new stream_id so
the tracker starts fresh and TrackEval still has GT. Exhausted pool → cycle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "mot_pipeline (Object_Detection + Tracking)"
sys.path[:0] = [str(PIPELINE), str(ROOT)]

from common import image_frames, load_sequence, read_detections  # noqa: E402
from slices import GUIDE_SEQUENCES, slice_dataset_root  # noqa: E402

NATIVE_FPS = 30.0
DPM_NAME = {
    "MOT17-02-FRCNN": "MOT17-02-DPM",
    "MOT17-04-FRCNN": "MOT17-04-DPM",
    "MOT17-09-FRCNN": "MOT17-09-DPM",
}


@dataclass(frozen=True)
class FrameReq:
    stream_id: str
    sequence: str
    frame_id: int
    path: Path
    width: int
    height: int
    replay: int


def test_sequences():
    root = slice_dataset_root("test")
    return [load_sequence(root, name) for name in GUIDE_SEQUENCES]


def stream_frames(max_frames=None, replays=1):
    """All frames of the test slice, then extra replays with distinct stream ids."""
    reqs = []
    for replay in range(replays):
        for sequence in test_sequences():
            for frame_id, path in image_frames(sequence, max_frames):
                reqs.append(FrameReq(
                    stream_id=f"{sequence.name}#r{replay}",
                    sequence=sequence.name,
                    frame_id=frame_id,
                    path=path,
                    width=sequence.width,
                    height=sequence.height,
                    replay=replay,
                ))
    return reqs


def cached_detections(tag, full_root=None):
    """Map test-slice frame ids → boxes.

    Prefers Guide v4 test-slice CSVs (already renumbered). Falls back to
    full-sequence DPM CSVs with the same frames, remapped onto 1..N.
    """
    from slices import load_manifest

    guide_root = PIPELINE / "outputs" / "guide" / "test" / "detections"
    full_root = Path(full_root or PIPELINE / "outputs" / "detections")
    manifest = load_manifest()
    out = {}
    for name in GUIDE_SEQUENCES:
        guide_src = guide_root / name / f"{tag}.csv"
        if guide_src.is_file():
            out[name] = read_detections(guide_src)
            continue
        start, end = manifest["test"][name]["original_frames"]
        src = full_root / DPM_NAME[name] / f"{tag}.csv"
        if not src.is_file():
            out[name] = {}
            continue
        grouped = read_detections(src)
        offset = start - 1
        remapped = {}
        for frame, boxes in grouped.items():
            if start <= frame <= end:
                remapped[frame - offset] = boxes
        out[name] = remapped
    return out


def gold_detections():
    from eval_harness import gold_keep, mot_to_detections
    out = {}
    for sequence in test_sequences():
        out[sequence.name] = mot_to_detections(sequence.directory / "gt" / "gt.txt", keep=gold_keep)
    return out
