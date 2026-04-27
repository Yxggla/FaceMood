from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceLandmarks:
    left_eye: tuple[int, int]
    right_eye: tuple[int, int]
    nose: tuple[int, int]
    mouth_left: tuple[int, int]
    mouth_right: tuple[int, int]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "nose": self.nose,
            "mouth_left": self.mouth_left,
            "mouth_right": self.mouth_right,
        }


class MediaPipeLandmarkDetector:
    def __init__(self, max_num_faces: int = 5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for landmark detection") from exc

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> FaceLandmarks | None:
        x1, y1, x2, y2 = _clip_bbox(bbox, frame.shape[1], frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        width = x2 - x1
        height = y2 - y1

        def point(index: int) -> tuple[int, int]:
            return (int(x1 + lm[index].x * width), int(y1 + lm[index].y * height))

        return FaceLandmarks(
            left_eye=point(33),
            right_eye=point(263),
            nose=point(1),
            mouth_left=point(61),
            mouth_right=point(291),
        )


class NullLandmarkDetector:
    def detect(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> FaceLandmarks | None:
        return None


def create_landmark_detector() -> MediaPipeLandmarkDetector | NullLandmarkDetector:
    try:
        return MediaPipeLandmarkDetector()
    except RuntimeError:
        return NullLandmarkDetector()


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

