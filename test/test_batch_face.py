import cv2
import numpy as np

from face_detection import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace()

    # Load the PNG image
    img = cv2.imread('assests/input_image.png')
    if img is None:
        print("Image not found or unable to load.")

    # Convert the image from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)

    # Detect faces in the image
    faces = detector(img)
    if len(faces) > 0:
        box, landmarks, score = faces[0]
        box = box.astype(np.int32)
        cv2.rectangle(
            img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2
        )

    # Display the image with the detected face
    cv2.imshow("Detected Face", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, img = cap.read()
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     print(img.shape)

    #     faces = detector(img)
    #     box, landmarks, score = faces[0]
    #     box = box.astype(np.int32)
    #     cv2.rectangle(
    #         img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2
    #     )
    #     cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(1)