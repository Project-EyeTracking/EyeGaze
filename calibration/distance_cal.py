import math

import cv2
import numpy as np


def calculate_focal_length():
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


def calculate_distance(focal_length):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # # Get focal length
    # focal_length = calculate_focal_length()

    # Known average face width in cm
    KNOWN_FACE_WIDTH = 16.0  # cm - average adult face width

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for x, y, w, h in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Calculate distance using triangle similarity
            # Distance = (known face width × focal length) / face width in pixels
            distance = (KNOWN_FACE_WIDTH * focal_length) / w

            # Display distance
            distance_text = f"Distance: {distance:.2f} cm"
            cv2.putText(
                frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        # Display the frame
        cv2.imshow("Distance Estimation", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    focal_length = calculate_focal_length()
    calculate_distance(focal_length)
