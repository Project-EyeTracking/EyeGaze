import csv
import pathlib
import time

import cv2
import numpy as np

from l2cs import render

CWD = pathlib.Path.cwd()


def process_video(
    video_path,
    gaze_estimator,
    draw_head_pose=False,
    draw_gaze=True,
    output_mode="visualize",
    mapper=None,
    ar_detector=None,
    marker_id=None,
):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    out = None
    if output_mode in ["save", "both"]:
        output_dir = CWD / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"processed_video_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # CSV setup
    csv_file_path = CWD / "output" / f"processed_coordinates_{int(time.time())}.csv"
    csv_file_path.parent.mkdir(exist_ok=True)
    csv_file = open(csv_file_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "Frame",
            "Time_sec",
            "Pitch_rad",
            "Yaw_rad",
            "Pitch_deg",
            "Yaw_deg",
            "ScreenX",
            "ScreenY",
            "Distance_cm",
        ]
    )

    no_gaze_count = 0
    frame_index = 0
    try:
        while cap.isOpened():
            # frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Get current time in video
            time_in_video = frame_index / fps

            # Perform gaze estimation
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
                    yaw_rad = results.yaw[0]
                    pitch_rad = results.pitch[0]
                    yaw_deg = np.degrees(yaw_rad)
                    pitch_deg = np.degrees(pitch_rad)

                    # Calculate screen coordinates
                    screen_x, screen_y = mapper.angles_to_screen_point(pitch_rad, yaw_rad)
                    distance_cm = mapper.screen.distance_cm

                    # Log data to CSV
                    csv_writer.writerow(
                        [
                            frame_index,
                            time_in_video,
                            pitch_rad,
                            yaw_rad,
                            pitch_deg,
                            yaw_deg,
                            screen_x,
                            screen_y,
                            distance_cm,
                        ]
                    )

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

            frame_index += 1
            # Exit on pressing 'q'
            if output_mode != "none" and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        csv_file.close()
        cap.release()
        if out:
            out.release()
            print(f"Processed video saved to: {output_path}")
        if output_mode in ["visualize", "both"]:
            cv2.destroyAllWindows()
