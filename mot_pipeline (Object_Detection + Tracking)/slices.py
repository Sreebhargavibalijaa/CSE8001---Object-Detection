"""Guide-defined MOT17 development / test slices (Pipeline Evaluation Guide v4).

Static-camera FRCNN folders only. Each sequence is split at half its seqinfo.ini
length. Frame IDs below are original MOT17 indices; trimmed copies renumber from 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import REPO_ROOT, load_sequence

# MOT17 train, FRCNN public-detection folders (identical frames to DPM/SDP).
GUIDE_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")

# seqLength from each sequence's seqinfo.ini. Development = [1, half],
# test = [half+1, seqLength], half = seqLength // 2.
SEQUENCE_LENGTHS = {
    "MOT17-02-FRCNN": 600,
    "MOT17-04-FRCNN": 1050,
    "MOT17-09-FRCNN": 525,
}

SLICE_ROOT = REPO_ROOT / "eval_slices"
MANIFEST_PATH = SLICE_ROOT / "manifest.json"
DATASET_VERSION_PATH = SLICE_ROOT / "DATASET_VERSION.txt"

RNG_SEED = 8001  # numpy.random.default_rng(8001); unused because the split is deterministic.


def half_length(seq_length):
    return seq_length // 2


def slice_window(seq_length, which):
    half = half_length(seq_length)
    if which == "development":
        return 1, half
    if which == "test":
        return half + 1, seq_length
    raise ValueError(which)


def trimmed_length(seq_length, which):
    start, end = slice_window(seq_length, which)
    return end - start + 1


def slice_dataset_root(which):
    return SLICE_ROOT / f"MOT17_{which}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_version(dataset_root, name):
    sequence = load_sequence(dataset_root, name)
    return {
        "name": name,
        "seq_length": sequence.seq_length,
        "frame_rate": sequence.frame_rate,
        "width": sequence.width,
        "height": sequence.height,
        "seqinfo_sha256": sha256_file(sequence.directory / "seqinfo.ini"),
        "gt_sha256": sha256_file(sequence.directory / "gt" / "gt.txt"),
        "det_sha256": sha256_file(sequence.directory / "det" / "det.txt"),
    }


def build_manifest(dataset_root):
    sequences = {}
    development = {}
    test = {}
    for name in GUIDE_SEQUENCES:
        info = sequence_version(dataset_root, name)
        if info["seq_length"] != SEQUENCE_LENGTHS[name]:
            raise ValueError(f"{name} seqLength {info['seq_length']} != {SEQUENCE_LENGTHS[name]}")
        sequences[name] = info
        length = info["seq_length"]
        for which, store in (("development", development), ("test", test)):
            start, end = slice_window(length, which)
            store[name] = {
                "original_frames": [start, end],
                "trimmed_frames": [1, end - start + 1],
                "trimmed_length": end - start + 1,
                "frame_ids": f"{start}..{end} inclusive ({end - start + 1} frames)",
            }
    return {
        "guide": "Pipeline Evaluation Guide v4",
        "dataset": "MOT17",
        "subset": "train",
        "folders": "FRCNN",
        "note": (
            "DPM/FRCNN/SDP folders share frames and GT; the guide requires the FRCNN "
            "folders. Split is first/second half of seqinfo.ini seqLength, not a random sample. "
            f"numpy.random.default_rng({RNG_SEED}) is recorded for reproducibility but is not used."
        ),
        "rng_seed": RNG_SEED,
        "sequences": sequences,
        "development": development,
        "test": test,
        "slice_roots": {
            "development": str(slice_dataset_root("development").relative_to(REPO_ROOT)),
            "test": str(slice_dataset_root("test").relative_to(REPO_ROOT)),
        },
    }


def load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing {MANIFEST_PATH}; run build_slices.py")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
