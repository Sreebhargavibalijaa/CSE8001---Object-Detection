"""Shared helpers: MOT17 sequence info and the detection / track file formats."""

import configparser
import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "MOT17"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Detections: original-image pixel coordinates, class_id 0 = person.
DETECTION_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "confidence", "class_id"]


@dataclass(frozen=True)
class Sequence:
    name: str
    directory: Path
    image_directory: Path
    frame_rate: float
    seq_length: int
    width: int
    height: int
    image_extension: str

    def frame_limit(self, max_frames=None):
        if max_frames is None:
            return self.seq_length
        if not 1 <= max_frames <= self.seq_length:
            raise ValueError(f"--max-frames must be between 1 and {self.seq_length}")
        return max_frames


def load_sequence(dataset_root, name):
    directory = Path(dataset_root).resolve() / "train" / name
    parser = configparser.ConfigParser()
    with (directory / "seqinfo.ini").open(encoding="utf-8") as file:
        parser.read_file(file)
    section = parser["Sequence"]
    return Sequence(
        name=section["name"], directory=directory,
        image_directory=directory / section["imDir"],
        frame_rate=section.getfloat("frameRate"), seq_length=section.getint("seqLength"),
        width=section.getint("imWidth"), height=section.getint("imHeight"),
        image_extension=section["imExt"],
    )


def image_frames(sequence, max_frames=None):
    """(frame id, image path) for frames 1..N; the numeric filename is the frame id."""
    frames = []
    for frame in range(1, sequence.frame_limit(max_frames) + 1):
        path = sequence.image_directory / f"{frame:06d}{sequence.image_extension}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing image frame: {path}")
        frames.append((frame, path))
    return frames


def add_common_arguments(parser):
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="MOT17 folder containing train/")
    parser.add_argument("--sequence", default="MOT17-02-DPM")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-frames", type=int, help="Only use frames 1 through N")


def write_detections(path, rows):
    """rows: (frame, x1, y1, x2, y2, confidence)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(DETECTION_COLUMNS)
        writer.writerows((*row, 0) for row in rows)


def read_detections(path):
    """frame -> list of [x1, y1, x2, y2, confidence, class_id]"""
    grouped = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        if next(reader, None) != DETECTION_COLUMNS:
            raise ValueError(f"Detection CSV header must be exactly {DETECTION_COLUMNS}")
        for values in reader:
            grouped.setdefault(int(float(values[0])), []).append([float(value) for value in values[1:7]])
    return grouped


def write_tracks(path, rows):
    """MOTChallenge format, no header: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows((*row, -1, -1, -1) for row in rows)


def read_tracks(path):
    """(frame, track id, left, top, width, height, confidence)"""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for values in csv.reader(file):
            if len(values) != 10:
                raise ValueError("MOT track files must have exactly 10 columns, without a header")
            rows.append((int(float(values[0])), int(float(values[1])), *map(float, values[2:7])))
    return rows
