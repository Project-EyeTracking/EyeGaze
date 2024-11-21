import math
from collections import deque

import cv2
import numpy as np


class ArUcoDistanceEstimator:
    def __init__(self):
        # Initialize face detector and ArUco detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Constants
        self.ARUCO_MARKER_SIZE = 2.0  # Size of ArUco marker in cm
        self.QUEUE_SIZE = 10
        self.distance_history = deque(maxlen=self.QUEUE_SIZE)

        # Distance filters
        self.MIN_DISTANCE = 30.0
        self.MAX_DISTANCE = 300.0
        self.MAX_DISTANCE_CHANGE = 50.0
        self.last_valid_distance = None

        # Kalman Filter setup
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

    def generate_aruco_marker(self):
        """Generate an ArUco marker image to be printed."""
        marker_id = 0
        marker_size = 200  # pixels
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        # marker_image = cv2.aruco.drawMarker(self.aruco_dict, marker_id, marker_size, marker_image, 1)
        marker_image = cv2.aruco.generateImageMarker(
            self.aruco_dict, marker_id, marker_size, marker_image, 1
        )

        # Save the marker
        cv2.imwrite("aruco_marker.png", marker_image)
        print(
            f"ArUco marker saved as 'aruco_marker.png'. Print it with size {self.ARUCO_MARKER_SIZE}cm x {self.ARUCO_MARKER_SIZE}cm"
        )
        return marker_image

    def detect_aruco_marker(self, frame):
        """Detect ArUco marker and return its corners."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            return corners[0][0]  # Return corners of first detected marker
        return None

    def calculate_marker_distance(self, corners, focal_length):
        """Calculate distance based on ArUco marker size."""
        if corners is None:
            return None

        # Calculate marker width in pixels
        marker_width_pixels = np.linalg.norm(corners[0] - corners[1])

        # Calculate distance using triangle similarity
        distance = (self.ARUCO_MARKER_SIZE * focal_length) / marker_width_pixels
        return distance

    def apply_filters(self, distance):
        """Apply all filtering mechanisms."""
        if distance is None:
            return None, None, None

        # Validate distance
        if distance < self.MIN_DISTANCE or distance > self.MAX_DISTANCE:
            return None, None, None

        if self.last_valid_distance is not None:
            if abs(distance - self.last_valid_distance) > self.MAX_DISTANCE_CHANGE:
                return None, None, None

        # Moving average
        self.distance_history.append(distance)
        smoothed_distance = np.mean(self.distance_history)

        # Kalman filter
        filtered_distance = self.kalman_filter(smoothed_distance)

        self.last_valid_distance = distance
        return distance, smoothed_distance, filtered_distance

    def kalman_filter(self, measurement):
        """Apply Kalman filter."""
        prediction = self.kalman.predict()
        if measurement is not None:
            correction = self.kalman.correct(np.array([[measurement]], np.float32))
            return float(correction[0])
        return float(prediction[0])

    def calculate_focal_length(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Get the frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Approximate horizontal and vertical FOV angles (in degrees)
        # These values are typical for most webcams but may need adjustment
        horizontal_fov = 60  # degrees
        vertical_fov = 45  # degrees

        # Convert FOV angles to radians
        horizontal_fov_rad = math.radians(horizontal_fov)
        vertical_fov_rad = math.radians(vertical_fov)

        # Calculate focal length using the formula: f = (image_width/2) / tan(FOV/2)
        focal_length_horizontal = (frame_width / 2) / math.tan(horizontal_fov_rad / 2)
        focal_length_vertical = (frame_height / 2) / math.tan(vertical_fov_rad / 2)

        # Average the two focal lengths
        focal_length = (focal_length_horizontal + focal_length_vertical) / 2

        # Release webcam
        cap.release()

        print(f"Frame dimensions: {frame_width}x{frame_height}")
        print(f"Estimated focal length: {focal_length:.2f} pixels")
        print(f"Focal length (horizontal): {focal_length_horizontal:.2f} pixels")
        print(f"Focal length (vertical): {focal_length_vertical:.2f} pixels")

        return focal_length

    def calculate_distance(self):
        cap = cv2.VideoCapture(0)
        focal_length = self.calculate_focal_length()  # Implement this method as before

        # # Generate marker first time
        # self.generate_aruco_marker()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect ArUco marker
            marker_corners = self.detect_aruco_marker(frame)

            if marker_corners is not None:
                # Calculate distance using marker
                raw_distance = self.calculate_marker_distance(marker_corners, focal_length)

                # Apply filters
                raw_distance, smoothed_distance, filtered_distance = self.apply_filters(
                    raw_distance
                )

                if filtered_distance is not None:
                    # Draw marker outline
                    cv2.polylines(frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)

                    # Display distances
                    y_offset = 30
                    if raw_distance:
                        cv2.putText(
                            frame,
                            f"Raw: {raw_distance:.1f} cm",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    if smoothed_distance:
                        cv2.putText(
                            frame,
                            f"Smoothed: {smoothed_distance:.1f} cm",
                            (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    if filtered_distance:
                        cv2.putText(
                            frame,
                            f"Filtered: {filtered_distance:.1f} cm",
                            (10, y_offset + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )

                    # Draw confidence indicator
                    confidence = min(len(self.distance_history) / self.QUEUE_SIZE, 1.0)
                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.1%}",
                        (10, y_offset + 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Distance Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = ArUcoDistanceEstimator()
    # estimator.generate_aruco_marker()
    estimator.calculate_distance()
