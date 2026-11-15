"""Materialize trimmed MOT17 development and test slices (Guide v4).

Creates eval_slices/MOT17_{development,test}/train/<seq>/ with images (hardlinks
or copies), GT and public FRCNN detections remapped to frames 1..N, and an
updated seqinfo.ini. Writes eval_slices/manifest.json with every frame ID and
dataset file hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from common import DEFAULT_DATASET_ROOT, load_sequence
from slices import (
    DATASET_VERSION_PATH,
    GUIDE_SEQUENCES,
    MANIFEST_PATH,
    SLICE_ROOT,
    build_manifest,
    slice_dataset_root,
    slice_window,
)


def remap_mot_file(src, dest, start, end, offset):
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            values = line.strip().split(",")
            frame = int(float(values[0]))
            if start <= frame <= end:
                values[0] = str(frame - offset)
                rows.append(",".join(values))
    dest.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def link_or_copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)


def write_seqinfo(sequence, dest, length):
    dest.write_text(
        "[Sequence]\n"
        f"name={sequence.name}\n"
        "imDir=img1\n"
        f"frameRate={int(sequence.frame_rate) if sequence.frame_rate == int(sequence.frame_rate) else sequence.frame_rate}\n"
        f"seqLength={length}\n"
        f"imWidth={sequence.width}\n"
        f"imHeight={sequence.height}\n"
        f"imExt={sequence.image_extension}\n",
        encoding="utf-8",
    )


def materialize(dataset_root, which, spec):
    root = slice_dataset_root(which) / "train"
    if root.exists():
        shutil.rmtree(root)
    for name in GUIDE_SEQUENCES:
        sequence = load_sequence(dataset_root, name)
        start, end = spec[name]["original_frames"]
        offset = start - 1
        dest = root / name
        dest.mkdir(parents=True)
        write_seqinfo(sequence, dest / "seqinfo.ini", spec[name]["trimmed_length"])
        remap_mot_file(sequence.directory / "gt" / "gt.txt", dest / "gt" / "gt.txt", start, end, offset)
        remap_mot_file(sequence.directory / "det" / "det.txt", dest / "det" / "det.txt", start, end, offset)
        for frame in range(start, end + 1):
            src = sequence.image_directory / f"{frame:06d}{sequence.image_extension}"
            if not src.is_file():
                raise FileNotFoundError(src)
            link_or_copy(src, dest / "img1" / f"{frame - offset:06d}{sequence.image_extension}")
        print(f"  {which}/{name}: original {start}-{end} -> {spec[name]['trimmed_length']} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build Guide v4 MOT17 development/test slices")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args()

    SLICE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.dataset_root)
    for which in ("development", "test"):
        print(f"==> {which}", flush=True)
        materialize(args.dataset_root, which, manifest[which])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "dataset: MOT17 train (FRCNN folders)",
        "guide: Pipeline Evaluation Guide v4",
        f"source: {Path(args.dataset_root).resolve()}",
        "split: first half of seqinfo.ini seqLength = development; remainder = test",
        "sequences:",
    ]
    for name, info in manifest["sequences"].items():
        lines.append(
            f"  {name}: seqLength={info['seq_length']} fps={info['frame_rate']} "
            f"gt={info['gt_sha256'][:12]} seqinfo={info['seqinfo_sha256'][:12]}"
        )
        dev, test = manifest["development"][name], manifest["test"][name]
        lines.append(f"    development original frames {dev['original_frames'][0]}-{dev['original_frames'][1]}")
        lines.append(f"    test         original frames {test['original_frames'][0]}-{test['original_frames'][1]}")
    DATASET_VERSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    print(f"wrote {DATASET_VERSION_PATH}")


if __name__ == "__main__":
    main()
