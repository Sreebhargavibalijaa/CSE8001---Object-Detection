from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo26n.pt")

# MOT17 train directory
base_path = "/Users/sreebhargavibalija/Desktop/8001/MOT17/train"

# Go through every MOT17 sequence
for sequence in sorted(os.listdir(base_path)):

    sequence_path = os.path.join(base_path, sequence)

    # Skip files
    if not os.path.isdir(sequence_path):
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

    # Get all JPG images
    images = sorted([
        f for f in os.listdir(img_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("Number of images:", len(images))

    # Process every image
    for image_name in images:

        image_file = os.path.join(img_path, image_name)

        print("Processing:", image_name)

        # Run YOLO
        results = model(image_file)

        for result in results:

            # Draw bounding boxes
            annotated_image = result.plot()

            # Save annotated image
            save_file = os.path.join(
                output_path,
                image_name
            )

            import cv2
            cv2.imwrite(save_file, annotated_image)

            print("Saved:", save_file)

print("\nDONE!")

