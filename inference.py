import os
import argparse
import pathlib
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
        os.environ['PYTORCH_MPS_SUPPORT'] = '1'
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
    parser = argparse.ArgumentParser(description='Gaze evaluation using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument('--device', dest='device', default="cpu", type=str, help='Device to run model: cpu or cuda or mps')
    parser.add_argument('--cam', dest='cam_id', default=0, type=int, help='Camera device id to use [0]')
    parser.add_argument('--video_path', dest='video_path', type=str, help='Input video path (optional)')
    parser.add_argument("--image_path", dest='image_path', type=str, help="Path to the input eye image (optional)")
    parser.add_argument("--model_path", dest='model_path', default='models/L2CSNet_gaze360.pkl', type=str, help="Path to the trained model (.pth file)")
    args = parser.parse_args()
    return args


def process_image(image, gaze_estimator):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Perform gaze estimation
    results = gaze_estimator.predict(image_np)

    print(f'Predicted gaze angles (pitch, yaw): {results.pitch, results.yaw}')

    # Visualize output
    frame = render(image_np, results)
    return frame


def process_webcam(cam_id, gaze_estimator):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {cam_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform gaze estimation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        # Visualize output
        output_frame = render(frame, results)

        cv2.imshow("Gaze Estimation", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    model_path = CWD / 'models' / 'L2CSNet_gaze360.pkl'
    # image_path = CWD / 'assets' / 'input_image.png'

    device = select_device()

    # Load the model
    model = load_model(model_path, device)

    # Create GazeEstimator instance
    gaze_estimator = GazeEstimator(model=model, device=device, include_detector=True, confidence_threshold=0.9)

    if args.image_path:
        # Load and prepare the image
        image = Image.open(args.image_path).convert('RGB')
        frame = process_image(image, gaze_estimator)
        cv2.imshow("Gaze Estimation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.video_path:
        # Process video file
        cap = cv2.VideoCapture(args.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Perform gaze estimation
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = gaze_estimator.predict(frame)

            # Visualize output
            output_frame = render(frame, results)
            cv2.imshow("Gaze Estimation", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # print(args.cam_id)
        # If neither image nor video is provided, use webcam
        process_webcam(args.cam_id, gaze_estimator)
