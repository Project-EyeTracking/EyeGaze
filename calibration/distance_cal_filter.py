import math
from collections import deque

import cv2
import numpy as np
import os
import json
import OC_calibration



class DistanceEstimator:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Constants
        self.KNOWN_FACE_WIDTH = 16.0  # cm
        self.QUEUE_SIZE = 10  # Number of frames for moving average
        self.distance_history = deque(maxlen=self.QUEUE_SIZE)

        # Distance filters
        self.MIN_DISTANCE = 30.0  # cm
        self.MAX_DISTANCE = 300.0  # cm
        self.MAX_DISTANCE_CHANGE = 50.0  # cm per frame
        self.last_valid_distance = None

        # Kalman Filter setup
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

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

    def moving_average_filter(self, distance):
        """Apply moving average filter to smooth distance measurements."""
        self.distance_history.append(distance)
        return np.mean(self.distance_history)

    def kalman_filter(self, measurement):
        """Apply Kalman filter to the measurement."""
        prediction = self.kalman.predict()
        if measurement is not None:
            correction = self.kalman.correct(np.array([[measurement]], np.float32))
            return float(correction[0])
        return float(prediction[0])

    def validate_distance(self, distance):
        """Validate and filter the calculated distance."""
        if distance is None:
            return None

        # Check if distance is within reasonable range
        if distance < self.MIN_DISTANCE or distance > self.MAX_DISTANCE:
            return self.last_valid_distance

        # Check for sudden large changes
        if self.last_valid_distance is not None:
            distance_change = abs(distance - self.last_valid_distance)
            if distance_change > self.MAX_DISTANCE_CHANGE:
                return self.last_valid_distance

        self.last_valid_distance = distance
        return distance


    def get_focal_length_from_json(self,json_file):
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            return None
        
        # Open and load the JSON file
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Extract the focal length
        focal_length = data.get("FocalLength", None)
        if focal_length is not None:
            return focal_length
        else:
            return None

    def calculate_distance(self):
        cap = cv2.VideoCapture(0)
        #focal_length = self.calculate_focal_length()
        #print(focal_length)
        
        # Specify the JSON file path
        cwd = os.getcwd()
        json_file = os.path.join(cwd, "calibration_output.json")

        # Get the focal length
        focal_length = self.get_focal_length_from_json(json_file)

        #print("FL from checkerboard",focal_length1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for x, y, w, h in faces:
                # Calculate raw distance
                raw_distance = (self.KNOWN_FACE_WIDTH * focal_length) / w

                # Apply filters
                validated_distance = self.validate_distance(raw_distance)
                if validated_distance is None:
                    continue

                # smoothed_distance = self.moving_average_filter(validated_distance)
                # filtered_distance = self.kalman_filter(smoothed_distance)

                # Draw rectangle and display distances
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display different distance measurements
                y_offset = y - 60
                cv2.putText(
                    frame,
                    f"Raw: {raw_distance:.1f} cm",
                    (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                # cv2.putText(
                #     frame,
                #     f"Smoothed: {smoothed_distance:.1f} cm",
                #     (x, y_offset + 20),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2,
                # )
                # cv2.putText(
                #     frame,
                #     f"Filtered: {filtered_distance:.1f} cm",
                #     (x, y_offset + 40),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 0, 0),
                #     2,
                # )

                # # Draw confidence indicator
                # confidence = min(len(self.distance_history) / self.QUEUE_SIZE, 1.0)
                # cv2.putText(
                #     frame,
                #     f"Confidence: {confidence:.1%}",
                #     (10, 30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2,
                # )

            # Stop processing after handling the first face
            

            cv2.imshow("Distance Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    OC_calibration.calculate_focal_length()
    estimator = DistanceEstimator()
    estimator.calculate_distance()
