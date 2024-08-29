import cv2
import numpy as np

from batch_face import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace(gpu_id=-1)

    img = cv2.imread("../assets/input_image.png")
    if img is None:
        print("Image not found or unable to load.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    max_size = 1080 # if the image's max size is larger than 1080, it will be resized to 1080, -1 means no resize
    resize = 1 # resize the image to speed up detection, default is 1, no resize
    threshold = 0.95 # confidence threshold

    # now we recommand to specify return_dict=True to get the result in a more readable way
    faces = detector(img, threshold=threshold, resize=resize, max_size=max_size, return_dict=True)
    
    print(f"Number of faces detected: {len(faces)}")

    for face in faces:
        box = face['box']
        kps = face['kps']
        score = face['score']
        print(f"Face: {box=}, {kps=}, {score=}")

        # Draw the bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the keypoints
        for i in range(5):
            x, y = int(kps[i][0]), int(kps[i][1])
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

        # Display the score
        score_text = f"Score: {score:.2f}"
        cv2.putText(img, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the image with all detected faces, bounding boxes, and scores
    cv2.imshow("Detected Faces", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
