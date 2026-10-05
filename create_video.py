
import os
import cv2

def build_video_from_images(
    image_dir="MOT17/train/MOT17-02-DPM/yolo_boxes",
    output_video_path="MOT17/train/MOT17-02-DPM/yolo_boxes_video.mp4",
    fps=30
):
    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print(f"No images found in {image_dir}")
        return

    first_image_path = os.path.join(image_dir, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Could not load image: {first_image_path}")

    height, width, _ = first_image.shape
    print(f"Found {len(images)} images.")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print(f"Writing video to: {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to load {img_path}, skipping.")
            continue
        out.write(frame)
        if idx % 50 == 0 or idx == len(images):
            print(f"Processed frame {idx}/{len(images)}")

    out.release()
    print(f"Video creation complete! Output saved to: {output_video_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert an image sequence to an MP4 video.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="MOT17/train/MOT17-02-DPM/yolo_boxes",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="MOT17/train/MOT17-02-DPM/yolo_boxes_video.mp4",
        help="Path to output MP4 video file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second"
    )

    args = parser.parse_args()
    build_video_from_images(
        image_dir=args.image_dir,
        output_video_path=args.output,
        fps=args.fps
    )
