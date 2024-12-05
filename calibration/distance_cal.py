import math
import cv2
import numpy as np

import OC_calibration  #importing to get the focal length


def calculate_distance(focal_length):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

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
    focal_length = OC_calibration.optical_centre()
    calculate_distance(focal_length)
