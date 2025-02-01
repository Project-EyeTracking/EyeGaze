import pathlib
import time

import cv2
import numpy as np

from l2cs import render

CWD = pathlib.Path.cwd()


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
