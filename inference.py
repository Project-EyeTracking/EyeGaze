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
        device = torch.device("cuda:0")
        print(f"Using CUDA: {torch.cuda.get_device_name(device)}")
        #print("id",device.index)
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
    #parser.add_argument(
    #    "--cam", dest="cam_id", type=int, help="Camera device id to use [0]"
    #)
    parser.add_argument(
        "--video_path", dest="video_path",default=0, type=str, help="Input video path (optional)"
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
    parser.add_argument(
        "--output",
        dest="output",
        choices=["visualize", "save", "both", "none"],
        default="visualize",
        help="Output mode: visualize, save, both, or none (default: visualize)",
    )
    args = parser.parse_args()
    return args


"""def process_image(
    image, gaze_estimator, draw_head_pose=True, draw_gaze=True, output_mode="visualize"
):
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

    if output_mode in ["save", "both"]:
        output_dir = CWD / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"processed_image_{int(time.time())}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Processed image saved to: {output_path}")

    if output_mode in ["visualize", "both"]:
        cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frame, results


def process_webcam(
    cam_id, gaze_estimator, draw_head_pose=True, draw_gaze=True, output_mode="visualize"
):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {cam_id}")
        return

    # # Get video properties
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = 30  # Set output FPS to 30

    # out = None
    # if output_mode in ["save", "both"]:
    #     output_dir = CWD / "output"
    #     output_dir.mkdir(exist_ok=True)
    #     output_path = output_dir / f"webcam_output_{int(time.time())}.mp4"
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

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
                frame = render(frame, results, draw_landmarks=False, draw_bboxes=True)

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

            # if out:
            #     out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # if output_mode in ["visualize", "both"]:
            #     cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        else:
            print("No gaze detected in this frame")
            no_gaze_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # if output_mode != "none" and cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    # if out:
    #     out.release()
    #     print(f"Webcam output saved to: {output_path}")
    # if output_mode in ["visualize", "both"]:
    #     cv2.destroyAllWindows()
"""

def process_video(video_path, gaze_estimator, draw_head_pose=False, draw_gaze=True, output_mode="save"):
    #print("here")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Default to 30 if FPS is not available

    
    # Screen parameters (adjust these according to your setup)
    screen_width_px = 1920    # Screen width in pixels
    screen_height_px = 1080   # Screen height in pixels
    screen_width_cm = 47.6    # Screen width in centimeters
    screen_height_cm = 26.8   # Screen height in centimeters
    screen_distance_cm = 60   # Distance from eyes to screen in centimeters

    out = None
    if output_mode in ["save", "both"]:
        output_dir = CWD / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"processed_video_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc,fps, (width, height))

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

            # Assuming single face
            pitch = results.pitch[0]
            yaw = results.yaw[0]

            # Convert angles from degrees to radians
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)

            # Calculate gaze point in centimeters relative to screen center
            x_cm = screen_distance_cm * np.tan(yaw_rad)
            y_cm = screen_distance_cm * np.tan(pitch_rad)

            # Map physical coordinates to pixel coordinates
            x_px = (x_cm / (screen_width_cm / 2)) * (screen_width_px / 2) + (screen_width_px / 2)
            y_px = -(y_cm / (screen_height_cm / 2)) * (screen_height_px / 2) + (screen_height_px / 2)

            # Ensure coordinates are within screen bounds
            x_px = int(np.clip(x_px, 0, screen_width_px - 1))
            y_px = int(np.clip(y_px, 0, screen_height_px - 1))

            # Draw gaze point on the frame
            cv2.circle(frame, (x_px, y_px), 10, (0, 0, 255), -1)

            # Optional: Print the gaze coordinates
            print(f"Gaze coordinates: ({x_px}, {y_px})")

      
            if draw_gaze:
                frame = render(frame, results, draw_landmarks=False, draw_bboxes=True)


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

            if out:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if output_mode in ["visualize", "both"]:
                cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        else:
            print("No gaze detected in this frame")

            no_gaze_count += 1

        # Exit on pressing 'q'
        if output_mode != "none" and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
        print(f"Processed video saved to: {output_path}")
    if output_mode in ["visualize", "both"]:
        cv2.destroyAllWindows()

    #print(results)
    #print(no_gaze_count)

if __name__ == "__main__":
    args = parse_args()
    model_path = CWD / "models" / "L2CSNet_gaze360.pkl"
    args.video_path = r'C:\Users\anagh\Videos\camera_recording.mp4'

    device = select_device()
    #print("index",device.index)

    # Load the model
    model = load_model(model_path, device)
    #print("abov1e")

    # Create GazeEstimator instance
    gaze_estimator = GazeEstimator(
        model=model,
        device=device,
        include_detector=True,
        confidence_threshold=0.9,
        include_head_pose=True,
    )

    #print("above")

    if args.image_path:
        # Load and prepare the image
        image = Image.open(args.image_path).convert("RGB")
        frame, _ = process_image(
            image, gaze_estimator, draw_head_pose=True, draw_gaze=True, output_mode=args.output
        )

    elif args.video_path:
        #print("here")
        # Process video file
        process_video(
            args.video_path,
            gaze_estimator,
            draw_head_pose=False,
            draw_gaze=True,
            output_mode=args.output,
        )
    else:
        pass
    """else:
        # print(args.cam_id)
        # If neither image nor video is provided, use webcam
        process_webcam(
            args.cam_id,
            gaze_estimator,
            draw_head_pose=True,
            draw_gaze=True,
            output_mode=args.output,
        )"""
