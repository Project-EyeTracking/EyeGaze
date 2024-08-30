import argparse
import pathlib
import time

import cv2
import torch

from l2cs import Pipeline, render, select_device

# from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Gaze evaluation using model pretrained with L2CS-Net on Gaze360."
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="Device to run model: cpu or gpu:0",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "--cam", dest="cam_id", help="Camera device id to use [0]", default=0, type=int
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        help="Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152",
        default="ResNet50",
        type=str,
    )
    parser.add_argument(
        "--video_path",
        dest="video_path",
        help="Input video path",
        default="assests/video.mp4",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    arch = args.arch
    cam = args.cam_id

    gaze_pipeline = Pipeline(
        weights=CWD / "models" / "L2CSNet_gaze360.pkl",
        arch="ResNet50",
        device=select_device(args.device),
    )

    cap = cv2.VideoCapture(args.video_path)
    # cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise OSError("Cannot open webcam")

    # Get the width, height, and frames per second (fps) of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object to save the output video
    output_path = "assests/output_video.mp4"  # Output file path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()
            start_fps = time.time()

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = gaze_pipeline.step(frame)

            # Visualize output
            frame = render(frame, results)

            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(
                frame,
                f"FPS: {myFPS:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Write the processed frame to the output video
            out.write(frame)

            # cv2.imshow("Demo",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # success,frame = cap.read()

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
