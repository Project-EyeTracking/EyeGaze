import time

import cv2
import numpy as np

from l2cs import render


def process_webcam(
    cam_id,
    gaze_estimator,
    draw_head_pose=False,
    draw_gaze=True,
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
        # print(f"{mapper.screen.distance_cm=}")

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
