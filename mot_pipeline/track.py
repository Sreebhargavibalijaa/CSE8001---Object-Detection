"""Step 2: associate saved detections across frames with ByteTrack or OC-SORT.

Reads a detections CSV from detect.py (the detector is not run again) and writes
<output-root>/tracks/<sequence>/<detector>_<tracker>.txt in MOTChallenge format.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from common import add_common_arguments, image_frames, load_sequence, read_detections, write_tracks


class FrameDetections:
    """The boxes interface the Ultralytics trackers expect, including empty frames."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).reshape(-1, 6)

    @property
    def xyxy(self):
        return self.values[:, :4]

    @property
    def xywh(self):
        return np.concatenate([(self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2,
                               self.xyxy[:, 2:] - self.xyxy[:, :2]], axis=1)

    @property
    def conf(self):
        return self.values[:, 4]

    @property
    def cls(self):
        return self.values[:, 5]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return FrameDetections(self.values[index])


def build_tracker(name, config_path=None):
    config = YAML.load(check_yaml(str(config_path or f"{name}.yaml")))
    if config.get("tracker_type") != name:
        raise ValueError(f"Tracker config must have tracker_type={name}")
    return TRACKER_MAP[name](args=IterableSimpleNamespace(**config))


def track_color(track_id):
    return tuple(64 + ((track_id * factor) % 192) for factor in (37, 67, 97))


def render_video(sequence, frames, tracks, output, label):
    grouped = {}
    for row in tracks:
        grouped.setdefault(row[0], []).append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), sequence.frame_rate,
                             (sequence.width, sequence.height))
    for frame, path in frames:
        image = cv2.imread(str(path))
        for _, track_id, left, top, width, height, _ in grouped.get(frame, []):
            color = track_color(track_id)
            cv2.rectangle(image, (round(left), round(top)), (round(left + width), round(top + height)), color, 2)
            cv2.putText(image, f"ID {track_id}", (round(left), max(20, round(top) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"{label} | {sequence.name} | frame {frame}/{len(frames)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(image)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Track saved detections; no detector is run")
    add_common_arguments(parser)
    parser.add_argument("--detections", type=Path, required=True, help="CSV written by detect.py")
    parser.add_argument("--tracker", choices=["bytetrack", "ocsort"], default="bytetrack")
    parser.add_argument("--tracker-config", type=Path, help="Optional YAML overriding the Ultralytics defaults")
    parser.add_argument("--video", type=Path, help="Also render the tracks to this MP4 file")
    args = parser.parse_args()

    sequence = load_sequence(args.dataset_root, args.sequence)
    limit = sequence.frame_limit(args.max_frames)
    detections = read_detections(args.detections)
    tracker = build_tracker(args.tracker, args.tracker_config)

    tracks = []
    for frame in range(1, limit + 1):
        # Every frame is passed in order, including frames without detections.
        outputs = tracker.update(FrameDetections(detections.get(frame, [])))
        for x1, y1, x2, y2, track_id, confidence, *_ in outputs:
            tracks.append((frame, int(track_id), float(x1), float(y1), float(x2 - x1), float(y2 - y1),
                           float(confidence)))

    stem = f"{args.detections.stem}_{args.tracker}"
    output = args.output_root / "tracks" / sequence.name / f"{stem}.txt"
    write_tracks(output, tracks)
    print(f"{len(tracks)} track rows, {len({row[1] for row in tracks})} track IDs "
          f"over {limit} frames -> {output}")
    if args.video:
        render_video(sequence, image_frames(sequence, limit), tracks, args.video,
                     f"{args.detections.stem} + {args.tracker}")
        print(f"Video -> {args.video}")


if __name__ == "__main__":
    main()
