import cv2
import numpy as np
import torch
import torch.nn as nn

from batch_face import RetinaFace
from torchvision import transforms
from .results import GazeResultContainer


class GazeEstimator:
    def __init__(self, model, device, include_detector=True, confidence_threshold=0.5):
        self.model = model
        self.device = device
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.detector = None
        if self.include_detector:
            if device.type == 'cpu' or device.type == "mps":
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

    def prep_input_numpy(self, img: np.ndarray):
        """Preparing a Numpy Array as input to L2CS-Net."""
        if len(img.shape) == 4:
            imgs = [self.transform(im) for im in img]
            img = torch.stack(imgs)
        else:
            img = self.transform(img)

        img = img.to(self.device)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        return img

    def predict_gaze(self, image):
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

            if isinstance(image, np.ndarray):
                img = self.prep_input_numpy(image)

            # Predict
            gaze_pitch, gaze_yaw = self.model(img)
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180

            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted

    def predict(self, image):
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        max_size = 1080
        resize = 1

        if self.include_detector:
            faces = self.detector(image, threshold=self.confidence_threshold, resize=resize, max_size=max_size, return_dict=True)
            if faces:
                for face in faces:
                    if face['score'] < self.confidence_threshold:
                        continue

                    x_min = max(int(face['box'][0]), 0)
                    y_min = max(int(face['box'][1]), 0)
                    x_max = int(face['box'][2])
                    y_max = int(face['box'][3])

                    img = image[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    bboxes.append(face['box'])
                    landmarks.append(face['kps'])
                    scores.append(face['score'])

                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                else:
                    pitch = yaw = np.empty((0, 1))
            else:
                pitch = yaw = np.empty((0, 1))
        else:
            pitch = yaw = np.empty((0, 1))

        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.array(bboxes) if bboxes else None,
            landmarks=np.array(landmarks) if landmarks else None,
            scores=np.array(scores) if scores else None
        )

        return results
