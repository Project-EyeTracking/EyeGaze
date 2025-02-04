import argparse
import os
import pathlib

import torch
import torchvision
from PIL import Image

from calibration import ArucoDetector, GazeMapper, read_screen_specs
from l2cs import L2CS, GazeEstimator
from processing import process_image, process_video, process_webcam
from gen_insights import load_data, apply_smoothing,  compute_metrics, plot_coordinates

from game import setup_screen, game

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


def load_model(model_path, device):
    """Load the pre-trained model."""
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
    parser.add_argument(
        "--output",
        dest="output",
        choices=["visualize", "save", "both", "none"],
        default="visualize",
        help="Output mode: visualize, save, both, or none (default: visualize)",
    )
    args = parser.parse_args()
    return args

def insights(file1, file2):


    game_coords, processed_coords = load_data(file1, file2)
    # print(f"{processed_coords=}")

    smoothed_coords = apply_smoothing(
        processed_coords[0], processed_coords[1], method="kalman", window_size=5
    )

    compute_metrics(game_coords, smoothed_coords)

    plot_coordinates(game_coords, processed_coords, smoothed_coords)
    
    
    
if __name__ == "__main__":
    args = parse_args()
    model_path = CWD / "models" / "L2CSNet_gaze360.pkl"
    # image_path = CWD / 'assets' / 'input_image.png'
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
    print("\n--- Visible Range ---")
    print(
        f"Max Yaw: ±{visible_range['max_yaw_rad']:.3f} rad (±{visible_range['max_yaw_deg']:.1f}°)"
    )
    print(
        f"Max Pitch: ±{visible_range['max_pitch_rad']:.3f} rad (±{visible_range['max_pitch_deg']:.1f}°)"
    )
    print(
        f"Screen dimensions: {visible_range['screen_width_cm']} x {visible_range['screen_height_cm']} cm"
    )
    print(
        f"Screen resolution: {visible_range['screen_width_px']} x {visible_range['screen_height_px']} pixels"
    )
    print(f"Distance from screen: {visible_range['distance_cm']} cm")
    print("----------------------")
    
    
    
    
    # setup_screen()
    # game_file, game_video_file = game()
    # args.video_path = game_video_file
    if args.image_path:
        # Load and prepare the image
        image = Image.open(args.image_path).convert("RGB")
        frame, _ = process_image(
            image, gaze_estimator, draw_head_pose=False, draw_gaze=True, output_mode=args.output
        )

    elif args.video_path:
        # Process video file
        inference_output = process_video(
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
        # print(args.cam_id)
        # If neither image nor video is provided, use webcam
        process_webcam(
            args.cam_id,
            gaze_estimator,
            draw_head_pose=False,
            draw_gaze=True,
            mapper=mapper,
            ar_detector=ar_uco_detector,
            marker_id=42,
        )



    
    #insights(game_file, inference_output)