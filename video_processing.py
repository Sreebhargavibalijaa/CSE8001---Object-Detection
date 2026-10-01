from ultralytics import YOLO
import csv
import os
import cv2

# Load YOLO model
model = YOLO("yolo26n.pt")

# MOT17 train directory (relative to this repo)
base_path = os.path.join("MOT17", "train")

# Sequences to process. Add more names here to extend.
sequences = ["MOT17-02-DPM"]

# COCO class id for "person" -- the only object class MOT17 annotates
PERSON_CLASS_ID = 0

# Go through every requested MOT17 sequence
for sequence in sequences:

    sequence_path = os.path.join(base_path, sequence)

    # Skip files
    if not os.path.isdir(sequence_path):
        print("Sequence not found, skipping:", sequence_path)
        continue

    # Actual image directory
    img_path = os.path.join(sequence_path, "img1")

    # Check img1 exists
    if not os.path.isdir(img_path):
        continue

    print("\n===================================")
    print("Processing:", sequence)
    print("Images:", img_path)
    print("===================================")

    # Create output directory
    output_path = os.path.join(sequence_path, "yolo_boxes")

    os.makedirs(output_path, exist_ok=True)

    # Raw person detections, consumed by the tracking scripts
    detections_csv = os.path.join(sequence_path, "yolo_detections.csv")

    # Get all JPG images
    images = sorted([
        f for f in os.listdir(img_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("Number of images:", len(images))
    print("Detections CSV:", detections_csv)

    with open(detections_csv, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "frame",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "class_id"
        ])

        # Process every image
        for image_name in images:

            image_file = os.path.join(img_path, image_name)

            # MOT17 names frames 000001.jpg, 000002.jpg, ... so the file
            # stem is the 1-based frame number used by the MOT format
            frame = int(os.path.splitext(image_name)[0])

            print("Processing:", image_name)

            # Run YOLO
            results = model(image_file, verbose=False)

            for result in results:

                # Draw bounding boxes
                annotated_image = result.plot()

                # Save annotated image
                save_file = os.path.join(
                    output_path,
                    image_name
                )

                cv2.imwrite(save_file, annotated_image)

                # Save raw person detections for the tracker
                if result.boxes is None:
                    continue

                boxes = result.boxes.cpu().numpy()

                for box, conf, cls in zip(
                    boxes.xyxy,
                    boxes.conf,
                    boxes.cls
                ):

                    class_id = int(cls)

                    # MOT17 only scores people
                    if class_id != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = box.tolist()

                    writer.writerow([
                        frame,
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        f"{float(conf):.4f}",
                        class_id
                    ])

    print("Saved detections to:", detections_csv)

print("\nDONE!")
