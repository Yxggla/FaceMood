from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6, min_face_size: int = 60) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for MediaPipe face detection") from exc

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            raise RuntimeError("Installed mediapipe package does not expose mp.solutions.face_detection")

        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=float(min_detection_confidence),
        )
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_face_size = int(max(1, min_face_size))

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_detection.process(rgb)
        if not result.detections:
            return []

        detections: list[FaceDetection] = []
        for det in result.detections:
            score = float(det.score[0]) if getattr(det, "score", None) else 0.0
            if score < self.min_detection_confidence:
                continue
            rbb = det.location_data.relative_bounding_box
            x1 = int(rbb.xmin * width)
            y1 = int(rbb.ymin * height)
            w = int(rbb.width * width)
            h = int(rbb.height * height)
            x2 = x1 + w
            y2 = y1 + h
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            detections.append(FaceDetection((x1, y1, x2, y2), score))
        return detections


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


def create_face_detector(
    detector: str = "auto",
    min_detection_confidence: float = 0.6,
    min_face_size: int = 60,
) -> MediaPipeFaceDetector | OpenCVFaceDetector:
    detector = (detector or "auto").lower().strip()
    if detector == "haar":
        return OpenCVFaceDetector()
    if detector in ("mediapipe", "auto"):
        try:
            return MediaPipeFaceDetector(
                min_detection_confidence=min_detection_confidence,
                min_face_size=min_face_size,
            )
        except Exception:
            return OpenCVFaceDetector()
    return OpenCVFaceDetector()

