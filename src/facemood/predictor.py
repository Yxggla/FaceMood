from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import crop_aligned_face
from .config import MODEL_PATH
from .emotion_model import create_emotion_recognizer
from .face_detector import OpenCVFaceDetector
from .landmarks import FaceLandmarks, create_landmark_detector


@dataclass(frozen=True)
class FacePrediction:
    bbox: tuple[int, int, int, int]
    landmarks: FaceLandmarks | None
    emotion: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "landmarks": self.landmarks.as_dict() if self.landmarks else {},
            "emotion": self.emotion,
            "confidence": self.confidence,
        }


class FaceMoodPredictor:
    def __init__(self, model_path=MODEL_PATH, device: str | None = None) -> None:
        self.face_detector = OpenCVFaceDetector()
        self.landmark_detector = create_landmark_detector()
        self.emotion_recognizer = create_emotion_recognizer(Path(model_path), device=device)

    def predict_frame(self, frame: np.ndarray) -> list[FacePrediction]:
        predictions: list[FacePrediction] = []
        for detection in self.face_detector.detect(frame):
            landmarks = self.landmark_detector.detect(frame, detection.bbox)
            face = crop_aligned_face(frame, detection.bbox, landmarks)
            if face is None:
                emotion, confidence = "unknown", 0.0
            else:
                emotion, confidence = self.emotion_recognizer.predict(face)
            predictions.append(FacePrediction(detection.bbox, landmarks, emotion, confidence))
        return predictions
