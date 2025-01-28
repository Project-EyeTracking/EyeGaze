import argparse
import os
import pathlib
import time
import csv
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from calibration import ArucoDetector, GazeMapper, ScreenSpecs, read_screen_specs
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
        "--cam", dest="cam_id", type=int, help="Camera device id to use [0]"
    )
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


def process_image(
    image, gaze_estimator, draw_head_pose=False, draw_gaze=True, output_mode="visualize"
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
    cam_id,
    gaze_estimator,
    draw_head_pose=False,
    draw_gaze=True,
    output_mode="visualize",
    mapper=None,
    ar_detector=None,
    marker_id=None,
):
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

        processed_frame, distances = ar_detector.process_frame(frame)
        mapper.screen.distance_cm = distances.get(marker_id, 0.0)
     

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Perform gaze estimation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        # Check if results.pitch is not empty
        if results.pitch is not None and len(results.pitch) > 0:
            try:
                yaw = results.yaw[0]
                pitch = results.pitch[0]

                x, y = mapper.angles_to_screen_point(pitch, yaw)

                print(
                    f"\nGaze angles (yaw={np.degrees(yaw):.1f}° [{yaw:.3f} rad], pitch={np.degrees(pitch):.1f}° [{pitch:.3f} rad])"
                )
                print(f"Screen point: ({x}, {y}) pixels")

                mapper.visualize_gaze_point(x, y)

            except ValueError as e:
                print(f"Error: {e}")

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

        else:
            print("No gaze detected in this frame")
            no_gaze_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_video(
    video_path,
    gaze_estimator,
    draw_head_pose=False,
    draw_gaze=True,
    output_mode="none",  # Disable visualization
    mapper=None,
    ar_detector=None,
    marker_id=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # CSV setup
    csv_file_path = CWD / "output" / f"{video_path}.csv"
    csv_file_path.parent.mkdir(exist_ok=True)  # Ensure the output directory exists
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Frame",
            "Time",
            "Pitch(rad)",
            "Yaw(rad)",
            "Pitch(deg)",
            "Yaw(deg)",
            "ScreenX",
            "ScreenY",
            "Distance(cm)",
        ])

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or error in reading frame.")
                break

            # Get current time in video
            time_in_video = frame_index / fps

            # Perform gaze estimation
            processed_frame,distances = ar_detector.process_frame(frame)
            mapper.screen.distance_cm = distances.get(marker_id, 0.0)

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Perform gaze estimation
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = gaze_estimator.predict(frame)

            # Check if gaze results are available
            if results.pitch is not None and len(results.pitch) > 0:
                try:
                    pitch_rad = results.pitch[0]
                    yaw_rad = results.yaw[0]
                    pitch_deg = np.degrees(pitch_rad)
                    yaw_deg = np.degrees(yaw_rad)

                    # Calculate screen coordinates
                    screen_x, screen_y = mapper.angles_to_screen_point(pitch_rad, yaw_rad)
                    distance_cm = mapper.screen.distance_cm

                    # Log data to CSV
                    csv_writer.writerow([
                        frame_index,
                        time_in_video,
                        pitch_rad,
                        yaw_rad,
                        pitch_deg,
                        yaw_deg,
                        screen_x,
                        screen_y,
                        distance_cm,
                    ])

                except ValueError:
                    # Skip if there's an error in processing gaze data
                    pass

            frame_index += 1

    cap.release()
    print(f"Gaze results saved to: {csv_file_path}")



if __name__ == "__main__":
    args = parse_args()
    model_path = CWD / "models" / "L2CSNet_gaze360.pkl"
    output_path = CWD / 'output'
    args.video_path = r'C:\Users\k67885\Documents\EyeGaze\output\GameVideo_Horizontal_Medium1738079657.avi' #make it dynamic
    screen_spec_path = CWD / "calibration" / "screen_spec.json"

    device = select_device()

    # Load the model
    model = load_model(model_path, device)
    print(f"{device=}")

    # Create GazeEstimator instance
    gaze_estimator = GazeEstimator(
        model=model,
        device=device,
        include_detector=True,
        confidence_threshold=0.9,
        include_head_pose=False,
    )

    # Screen specifications
    screen = read_screen_specs(screen_spec_path)
    # print(f"{screen=}")

    ar_uco_detector = ArucoDetector(
        marker_real_width=4.0, calibration_file="calibration/calibration_results.json"
    )
    # Initialize mapper
    mapper = GazeMapper(screen)

    # Print visible range info
    visible_range = mapper.get_visible_range()
    # print("\n--- Visible Range ---")
    # print(
    #     f"Max Yaw: ±{visible_range['max_yaw_rad']:.3f} rad (±{visible_range['max_yaw_deg']:.1f}°)"
    # )
    # print(
    #     f"Max Pitch: ±{visible_range['max_pitch_rad']:.3f} rad (±{visible_range['max_pitch_deg']:.1f}°)"
    # )
    # print(
    #     f"Screen dimensions: {visible_range['screen_width_cm']}x{visible_range['screen_height_cm']} cm"
    # )
    # print(
    #     f"Screen resolution: {visible_range['screen_width_px']}x{visible_range['screen_height_px']} pixels"
    # )
    # print(f"Distance from screen: {visible_range['distance_cm']} cm")
    # print("----------------------")

    if args.image_path:
        # Load and prepare the image
        image = Image.open(args.image_path).convert("RGB")
        frame, _ = process_image(
            image, gaze_estimator, draw_head_pose=False, draw_gaze=True, output_mode=args.output
        )

    elif args.video_path:
        # Process video file
        process_video(
            args.video_path,
            gaze_estimator,
            draw_head_pose=False,
            draw_gaze=True,
            output_mode=args.output,
            mapper=mapper,
            ar_detector=ar_uco_detector,
            marker_id=42,
        )
    else:
        process_webcam(
            args.cam_id,
            gaze_estimator,
            draw_head_pose=False,
            draw_gaze=True,
            output_mode=args.output,
            mapper=mapper,
            ar_detector=ar_uco_detector,
            marker_id=42,
        )