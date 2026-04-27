from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


class OpenCVFaceDetector:
    """Lightweight baseline detector using OpenCV's bundled Haar cascade."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(40, 40),
        )
        detections: list[FaceDetection] = []
        for x, y, w, h in faces:
            detections.append(FaceDetection((int(x), int(y), int(x + w), int(y + h)), 1.0))
        return detections

