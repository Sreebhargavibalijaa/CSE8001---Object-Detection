import os
import cv2
import torch
import torchvision

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


# ============================================================
# LOAD FASTER R-CNN RESNET50 FPN
# ============================================================

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=weights
)

# IMPORTANT: inference mode
model.eval()


# ============================================================
# MOT17 TRAIN DIRECTORY
# ============================================================

base_path = "/Users/sreebhargavibalija/Desktop/8001/MOT17/train"


# ============================================================
# PROCESS EACH MOT17 SEQUENCE
# ============================================================

for sequence in sorted(os.listdir(base_path)):

    sequence_path = os.path.join(base_path, sequence)

    # Skip files
    if not os.path.isdir(sequence_path):
        continue

    # img1 directory
    img_path = os.path.join(sequence_path, "img1")

    if not os.path.isdir(img_path):
        continue

    print("\n===================================")
    print("Processing:", sequence)
    print("Images:", img_path)
    print("===================================")

    # Output directory
    output_path = os.path.join(
        sequence_path,
        "fb_resnet_boxes"
    )

    os.makedirs(output_path, exist_ok=True)

    # Get images
    images = sorted([
        f for f in os.listdir(img_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    print("Number of images:", len(images))


    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_name in images:

        image_file = os.path.join(
            img_path,
            image_name
        )

        print("Processing:", image_name)

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(image_file)

        if image is None:
            print("Could not read:", image_file)
            continue

        # OpenCV BGR -> RGB
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert image to PyTorch tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image_rgb
        ).permute(2, 0, 1).float() / 255.0

        # Faster R-CNN expects a LIST of tensors
        inputs = [image_tensor]


        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        with torch.no_grad():

            results = model(inputs)


        # Faster R-CNN returns a list
        result = results[0]

        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]


        # ----------------------------------------------------
        # Draw bounding boxes
        # ----------------------------------------------------

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            # Confidence threshold
            if score < 0.50:
                continue

            # Convert coordinates to integers
            x1, y1, x2, y2 = box.int().tolist()

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            text = f"{label.item()} {score.item():.2f}"

            cv2.putText(
                image,
                text,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # Save annotated image
        # ----------------------------------------------------

        save_file = os.path.join(
            output_path,
            image_name
        )

        cv2.imwrite(
            save_file,
            image
        )

        print("Saved:", save_file)


print("\nDONE!")
