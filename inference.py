import argparse
import os
import pathlib
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from l2cs import L2CS, GazeEstimator, render

CWD = pathlib.Path.cwd()


def select_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MPS is available for Apple Silicon (macOS)
        os.environ["PYTORCH_MPS_SUPPORT"] = "1"
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        # CUDA is available (GPU)
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(device)}")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        print("Using CPU")

    return device


# Load the pre-trained model
def load_model(model_path, device):
    # Create L2CS model
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Gaze evaluation using model pretrained with L2CS-Net on Gaze360."
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cpu",
        type=str,
        help="Device to run model: cpu or cuda or mps",
    )
    parser.add_argument(
        "--cam", dest="cam_id", default=0, type=int, help="Camera device id to use [0]"
    )
    parser.add_argument(
        "--video_path", dest="video_path", type=str, help="Input video path (optional)"
    )
    parser.add_argument(
        "--image_path", dest="image_path", type=str, help="Path to the input eye image (optional)"
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default="models/L2CSNet_gaze360.pkl",
        type=str,
        help="Path to the trained model (.pth file)",
    )
    args = parser.parse_args()
    return args


def process_image(image, gaze_estimator, draw_head_pose=False, draw_gaze=True):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Perform gaze estimation
    results = gaze_estimator.predict(image_np)

    print(f"Predicted gaze angles (pitch, yaw): {results.pitch, results.yaw}")

    frame = image_np.copy()

    # Visualize output based on arguments
    if draw_gaze:
        frame = render(frame, results, draw_landmarks=True, draw_bboxes=True)

    if draw_head_pose:
        for bbox, head_orientation in zip(results.bboxes, results.head_orientations):
            gaze_estimator.head_pose_estimator.plot_pose_cube(frame, bbox, **head_orientation)

    return frame, results


def process_webcam(cam_id, gaze_estimator, draw_head_pose=False, draw_gaze=True):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {cam_id}")
        return

    # FPS calculation variables
    frame_times = []
    fps = 0
    no_gaze_count = 0

    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform gaze estimation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        # Check if results.pitch is not empty
        if results.pitch is not None and len(results.pitch) > 0:
            # Visualize output
            if draw_gaze:
                frame = render(frame, results, draw_landmarks=True, draw_bboxes=True)

            if draw_head_pose:
                for bbox, head_orientation in zip(results.bboxes, results.head_orientations):
                    gaze_estimator.head_pose_estimator.plot_pose_cube(
                        frame, bbox, **head_orientation
                    )

            # Calculate and display FPS
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)

            # Calculate FPS over last 30 frames
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times)

            cv2.putText(
                frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"No Gaze Frames: {no_gaze_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        else:
            print("No gaze detected in this frame")
            no_gaze_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path, gaze_estimator, draw_head_pose=False, draw_gaze=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # FPS calculation variables
    frame_times = []
    fps = 0
    no_gaze_count = 0

    while cap.isOpened():
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform gaze estimation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        # Check if results.pitch is not empty
        if results.pitch is not None and len(results.pitch) > 0:
            # Visualize output
            if draw_gaze:
                frame = render(frame, results, draw_landmarks=True, draw_bboxes=True)

            if draw_head_pose:
                for bbox, head_orientation in zip(results.bboxes, results.head_orientations):
                    gaze_estimator.head_pose_estimator.plot_pose_cube(
                        frame, bbox, **head_orientation
                    )

            # Calculate and display FPS
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)

            # Calculate FPS over last 30 frames
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times)

            cv2.putText(
                frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"No Gaze Frames: {no_gaze_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        else:
            print("No gaze detected in this frame")
            no_gaze_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    model_path = CWD / "models" / "L2CSNet_gaze360.pkl"
    # image_path = CWD / 'assets' / 'input_image.png'

    device = select_device()

    # Load the model
    model = load_model(model_path, device)

    # Create GazeEstimator instance
    gaze_estimator = GazeEstimator(
        model=model,
        device=device,
        include_detector=True,
        confidence_threshold=0.9,
        include_head_pose=True,
    )

    if args.image_path:
        # Load and prepare the image
        image = Image.open(args.image_path).convert("RGB")
        frame, _ = process_image(
            image, gaze_estimator, draw_head_pose=gaze_estimator.include_head_pose, draw_gaze=False
        )

        cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.video_path:
        # Process video file
        process_video(
            args.video_path,
            gaze_estimator,
            draw_head_pose=gaze_estimator.include_head_pose,
            draw_gaze=False,
        )
    else:
        # print(args.cam_id)
        # If neither image nor video is provided, use webcam
        process_webcam(
            args.cam_id,
            gaze_estimator,
            draw_head_pose=gaze_estimator.include_head_pose,
            draw_gaze=False,
        )
