import cv2
from batch_face import drawLandmark_multiple, LandmarkPredictor, RetinaFace

if __name__ == "__main__":
    predictor = LandmarkPredictor(gpu_id=-1)
    detector = RetinaFace(gpu_id=-1)

    imgname = "../assets/input_image.png"
    img = cv2.imread(imgname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector(img, threshold=0.95)
    print(f"{faces=}")

    if len(faces) == 0:
        print("No face detected!")
        raise ValueError("No face detected in the image")

    # the first input for the predictor is a list of face boxes. [[x1,y1,x2,y2]]
    results = predictor(faces, img, from_fd=True) # from_fd=True to convert results from our detection results to simple boxes

    for face, landmarks in zip(faces, results):
        img = drawLandmark_multiple(img, face[0], landmarks)

    # Display or save the image with drawn landmarks
    cv2.imshow("Face with landmarks", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
