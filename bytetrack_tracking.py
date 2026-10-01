"""Read `<sequence>/yolo_detections.csv` produced by
`video_processing.py`, associate the boxes frame by frame, draw stable track
IDs on the original MOT17 images and write MOT-format results.
Swapping the tracker is a flag: `--tracker ocsort.yaml` gives OC-SORT, since
both trackers take the same detection interface. Swapping the detector is a
flag too: point `--detections` at any CSV with the same columns.
"""

import argparse
import csv
import os

import cv2
import numpy as np

from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


class Detections:
    """One frame of detections, shaped like the slice of ultralytics `Boxes`
    that the trackers actually touch: `xywh`, `conf`, `cls`, `len()` and
    indexing by a boolean mask.
    """

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @property
    def xyxy(self):
        xy = self.xywh[:, :2]
        wh = self.xywh[:, 2:4]
        return np.concatenate([xy - wh / 2, xy + wh / 2], axis=-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


def empty_detections():
    """Detections for a frame where the detector found nothing."""
    return Detections(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )


def load_detections(csv_path):
    """Read a detections CSV and group it by frame number.

    Expects the columns written by `video_processing.py`:
    frame,x1,y1,x2,y2,confidence,class_id
    """
    rows_by_frame = {}

    with open(csv_path, newline="") as csv_file:

        for row in csv.DictReader(csv_file):

            frame = int(row["frame"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            # The trackers want center-x, center-y, width, height
            rows_by_frame.setdefault(frame, []).append([
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
                float(row["confidence"]),
                float(row["class_id"]),
            ])

    detections_by_frame = {}

    for frame, rows in rows_by_frame.items():
        values = np.array(rows, dtype=np.float32)
        detections_by_frame[frame] = Detections(
            values[:, 0:4],
            values[:, 4],
            values[:, 5],
        )

    return detections_by_frame


def build_tracker(tracker_yaml):
    """Instantiate an ultralytics tracker from its YAML config."""
    config = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))

    if config.tracker_type not in TRACKER_MAP:
        raise ValueError(
            f"Unsupported tracker '{config.tracker_type}'. "
            f"Choose one of {sorted(TRACKER_MAP)}."
        )

    return TRACKER_MAP[config.tracker_type](args=config)


def color_for_id(track_id):
    """A stable, visually distinct BGR color per track ID.

    Steps the hue by the golden ratio so that consecutive IDs land far apart
    on the color wheel instead of shading into each other.
    """
    hue = int((int(track_id) * 0.618033988749895 * 180) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def draw_tracks(image, tracks):
    """Draw the box and ID of every track onto the frame."""
    for x1, y1, x2, y2, track_id, score in tracks:

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = color_for_id(track_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"ID {int(track_id)} {score:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


def main():

    parser = argparse.ArgumentParser(
        description="Track pre-computed detections with ByteTrack."
    )
    parser.add_argument(
        "--sequence_dir",
        type=str,
        default=os.path.join("MOT17", "train", "MOT17-02-DPM"),
        help="MOT17 sequence folder containing img1/",
    )
    parser.add_argument(
        "--detections",
        type=str,
        default=None,
        help="Detections CSV (default: <sequence_dir>/yolo_detections.csv)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config, e.g. bytetrack.yaml or ocsort.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write tracked frames (default: <sequence_dir>/<tracker>_boxes)",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="MOT-format results file (default: <sequence_dir>/<tracker>_results.txt)",
    )

    args = parser.parse_args()

    # Name outputs after the tracker so ByteTrack and OC-SORT runs do not
    # overwrite each other
    tracker_name = os.path.splitext(os.path.basename(args.tracker))[0]

    sequence_dir = args.sequence_dir
    img_path = os.path.join(sequence_dir, "img1")

    detections_csv = args.detections or os.path.join(
        sequence_dir, "yolo_detections.csv"
    )
    output_dir = args.output_dir or os.path.join(
        sequence_dir, f"{tracker_name}_boxes"
    )
    output_txt = args.output_txt or os.path.join(
        sequence_dir, f"{tracker_name}_results.txt"
    )

    if not os.path.isdir(img_path):
        raise SystemExit(f"No img1/ directory under {sequence_dir}")

    if not os.path.isfile(detections_csv):
        raise SystemExit(
            f"Detections not found: {detections_csv}\n"
            "Run video_processing.py first."
        )

    os.makedirs(output_dir, exist_ok=True)

    print("\n===================================")
    print("Tracker:", tracker_name)
    print("Sequence:", sequence_dir)
    print("Detections:", detections_csv)
    print("Tracked frames:", output_dir)
    print("MOT results:", output_txt)
    print("===================================")

    detections_by_frame = load_detections(detections_csv)
    tracker = build_tracker(args.tracker)

    images = sorted([
        f for f in os.listdir(img_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("Number of images:", len(images))

    total_tracks = 0

    with open(output_txt, "w", newline="") as txt_file:

        writer = csv.writer(txt_file)

        for image_name in images:

            # MOT17 names frames 000001.jpg, 000002.jpg, ... so the file stem
            # is the 1-based frame number shared with the CSV and the MOT file
            frame = int(os.path.splitext(image_name)[0])

            image_file = os.path.join(img_path, image_name)
            image = cv2.imread(image_file)

            if image is None:
                print("Could not read:", image_file)
                continue

            # Every frame must reach the tracker, even an empty one, so the
            # Kalman predictions and the lost-track buffer stay in step with
            # the real frame numbers
            frame_detections = detections_by_frame.get(
                frame, empty_detections()
            )

            # (N, 8) rows of [x1, y1, x2, y2, track_id, score, cls, idx]
            outputs = tracker.update(frame_detections)

            tracks = []

            if len(outputs):
                tracks = outputs[:, [0, 1, 2, 3, 4, 5]]

            for x1, y1, x2, y2, track_id, score in tracks:

                width = x2 - x1
                height = y2 - y1

                # MOT format: frame,id,x,y,width,height,confidence,-1,-1,-1
                writer.writerow([
                    frame,
                    int(track_id),
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{width:.2f}",
                    f"{height:.2f}",
                    f"{score:.4f}",
                    -1,
                    -1,
                    -1,
                ])

            total_tracks += len(tracks)

            annotated_image = draw_tracks(image, tracks)

            cv2.imwrite(os.path.join(output_dir, image_name), annotated_image)

            print(f"Frame {frame}: {len(tracks)} tracks")

    print("\nTotal track rows written:", total_tracks)
    print("Tracked frames saved to:", output_dir)
    print("MOT results saved to:", output_txt)
    print("\nDONE!")


if __name__ == "__main__":
    main()
