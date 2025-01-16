import json
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class ArucoDetector:
    """A class to handle ArUco marker detection and distance measurement."""

    def __init__(
        self,
        dictionary_id: int = cv2.aruco.DICT_6X6_250,
        marker_real_width: float = 4.0,
        calibration_file: str = "calibration_results.json",
    ):
        """Initialize the ArUco detector.

        Args:
            dictionary_id: The ID of the ArUco dictionary to use
            marker_real_width: The real-world width of the marker in centimeters
            calibration_file: Path to the camera calibration file
        """
        self.dictionary_id = dictionary_id
        self.marker_real_width = marker_real_width

        # Create ArUco dictionary and detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.focal_length = self._load_calibration(calibration_file)

    def _load_calibration(self, calibration_file: str) -> float:
        """Load camera calibration parameters from a JSON file.

        Args:
            calibration_file: Path to the calibration file

        Returns:
            float: Focal length from calibration file or None if loading fails
        """
        try:
            file_path = os.path.join(os.getcwd(), calibration_file)
            with open(file_path) as file:
                data = json.load(file)
            return data["focalLength"]["average"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading calibration file: {e}")
            return None

    @staticmethod
    def generate_marker(marker_id: int, marker_size: int, output_filename: str) -> None:
        """Generate an ArUco marker and save it to a file.

        Args:
            marker_id: ID of the marker to generate
            marker_size: Size of the marker in pixels
            output_filename: Output file path
        """
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_image, 1)
            marker_image = cv2.aruco.generateImageMarker(
                aruco_dict, marker_id, marker_size, marker_image, 1
            )

            cv2.imwrite(output_filename, marker_image)
            print(f"Marker generated successfully: {output_filename}")
        except Exception as e:
            print(f"Error generating marker: {e}")

    @staticmethod
    def calculate_distance(
        marker_corners: np.ndarray, marker_real_width: float, focal_length: float
    ) -> float:
        """Calculate the distance to a marker using its apparent size.

        Args:
            marker_corners: Corner coordinates of the detected marker
            marker_real_width: Real-world width of the marker in centimeters
            focal_length: Focal length of the camera

        Returns:
            float: Calculated distance in centimeters
        """
        if focal_length is None:
            return 0.0

        # Calculate the apparent width in pixels
        pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])

        # Use the distance formula: distance = (real_width * focal_length) / pixel_width
        distance = (marker_real_width * focal_length) / pixel_width
        return distance

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, float]]:
        """Process a single frame to detect markers and calculate distances.

        Args:
            frame: Input frame to process

        Returns:
            Tuple containing the processed frame and a dictionary of marker IDs and their distances
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers using the detector instance
        corners, ids, rejected = self.detector.detectMarkers(gray)

        distances = {}
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_corners in enumerate(corners):
                distance = self.calculate_distance(
                    marker_corners[0], self.marker_real_width, self.focal_length
                )
                marker_id = ids[i][0]
                distances[marker_id] = distance

                # Display distance on frame
                top_left = tuple(marker_corners[0][0].astype(int))
                cv2.putText(
                    frame,
                    f"Distance: {distance:.2f} cm",
                    top_left,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return frame, distances

    def run_detection(self, camera_id: int = 0) -> None:
        """Run continuous detection on camera feed.

        Args:
            camera_id: ID of the camera to use
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                processed_frame, distances = self.process_frame(frame)
                cv2.imshow("ArUco Distance Measurement", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function to demonstrate usage."""
    # Example usage
    detector = ArucoDetector(marker_real_width=4.0)

    # # Generate a marker
    # detector.generate_marker(42, 300, "aruco_marker_42.png")

    # Run detection
    detector.run_detection(camera_id=0)


if __name__ == "__main__":
    main()
