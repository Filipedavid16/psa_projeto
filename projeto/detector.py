import cv2
import numpy as np

from config import PROTO_PATH, MODEL_PATH, CONFIDENCE_THRESHOLD
from utils import get_center


class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        detected_faces = []

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            center = get_center(x1, y1, x2, y2)

            detected_faces.append({
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "confidence": conf
            })

        return detected_faces