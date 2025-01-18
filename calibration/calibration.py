import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraCalibrator:
    def __init__(
        self,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0,
        criteria: Tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    ):
        """Initialize the camera calibrator.

        Args:
            pattern_size: Tuple of (rows, cols) of the checkerboard pattern
            square_size: Size of each square in the checkerboard (in units you want to use)
            criteria: Termination criteria for cornerSubPix
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.criteria = criteria

        # Initialize object points
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

    def capture_calibration_images(
        self, cam_id: int = 0, output_dir: str = "calibration_images", max_images: int = 12
    ) -> str:
        """Capture images for calibration from webcam.

        Args:
            cam_id: Camera ID
            output_dir: Directory to save captured images
            max_images: Maximum number of images to capture

        Returns:
            str: Path to the output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not cap.isOpened():
            raise RuntimeError("Could not access the webcam.")

        image_count = 0
        print("Press 'c' to capture an image, or 'q' to quit.")

        try:
            while image_count < max_images:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame.")

                cv2.imshow("Webcam", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    image_path = os.path.join(output_dir, f"image_{image_count + 1}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Captured and saved: {image_path}")
                    image_count += 1
                elif key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"Captured {image_count} images. Images saved in '{output_dir}' directory.")
        return output_dir

    def find_corners(self, image: NDArray) -> Tuple[bool, Optional[NDArray]]:
        """Find chessboard corners in an image.

        Args:
            image: Input image

        Returns:
            Tuple of (success flag, corners if found)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners
        return False, None

    def calculate_reprojection_error(
        self,
        obj_points: List[NDArray],
        img_points: List[NDArray],
        rvecs: List[NDArray],
        tvecs: List[NDArray],
        camera_matrix: NDArray,
        dist_coeffs: NDArray,
    ) -> float:
        """Calculate the mean reprojection error."""
        total_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        return total_error / len(obj_points)

    def calibrate(
        self, image_dir: str
    ) -> Tuple[
        Optional[NDArray], Optional[NDArray], Optional[Tuple[float, float]], Optional[float]
    ]:
        """Perform camera calibration using images in the specified directory.

        Args:
            image_dir: Directory containing calibration images

        Returns:
            Tuple of (camera_matrix, distortion_coefficients, optical_center, reprojection_error)
        """
        obj_points = []
        img_points = []

        for filename in os.listdir(image_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image: {img_path}")
                continue

            ret, corners = self.find_corners(img)
            if ret:
                obj_points.append(self.objp)
                img_points.append(corners)
                cv2.drawChessboardCorners(img, self.pattern_size, corners, ret)
                cv2.imshow("Detected Corners", img)
                cv2.waitKey(500)
            else:
                print(f"Warning: No corners found in image: {img_path}")

        if not obj_points:
            print("Error: No valid calibration data found.")
            return None, None, None, None

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img.shape[:2][::-1], None, None
        )

        if not ret:
            print("Error: Camera calibration failed.")
            return None, None, None, None

        error = self.calculate_reprojection_error(
            obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs
        )

        optical_center = (camera_matrix[0, 2], camera_matrix[1, 2])
        focal_lengths = (camera_matrix[0, 0], camera_matrix[1, 1])

        print(f"Focal Lengths: fx = {focal_lengths[0]}, fy = {focal_lengths[1]}")
        return camera_matrix, dist_coeffs, optical_center, error

    def save_calibration_results(
        self,
        output_path: str,
        camera_matrix: Optional[NDArray],
        dist_coeffs: Optional[NDArray],
        optical_center: Optional[Tuple[float, float]],
        error: Optional[float],
    ) -> None:
        """Save calibration results to a JSON file."""
        if camera_matrix is None:
            output_data = {"error": "Calibration failed - no valid data"}
        else:
            output_data = {
                "cameraMatrix": camera_matrix.tolist(),
                "distortionCoefficients": dist_coeffs.flatten().tolist(),
                "opticalCenter": {"cx": float(optical_center[0]), "cy": float(optical_center[1])},
                "focalLength": {
                    "fx": float(camera_matrix[0, 0]),
                    "fy": float(camera_matrix[1, 1]),
                    "average": float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2),
                },
                "reprojectionError": float(error),
            }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)


def main():
    # Initialize calibrator with default settings
    calibrator = CameraCalibrator(pattern_size=(4, 7))

    # Capture calibration images
    image_dir = calibrator.capture_calibration_images(
        cam_id=0,
        output_dir="calibration_images",
        max_images=25
    )

    # Perform calibration
    image_dir = "calibration_images"
    camera_matrix, dist_coeffs, optical_center, error = calibrator.calibrate(image_dir)

    # Save results
    calibrator.save_calibration_results(
        "calibration_results.json", camera_matrix, dist_coeffs, optical_center, error
    )


if __name__ == "__main__":
    main()
