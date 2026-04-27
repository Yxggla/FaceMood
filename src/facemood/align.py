from __future__ import annotations

import math

import cv2
import numpy as np

from .config import IMAGE_SIZE
from .landmarks import FaceLandmarks


def crop_aligned_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: FaceLandmarks | None = None,
    output_size: int = IMAGE_SIZE,
) -> np.ndarray | None:
    x1, y1, x2, y2 = _expand_bbox(bbox, frame.shape[1], frame.shape[0], scale=0.18)
    if x2 <= x1 or y2 <= y1:
        return None

    working = frame
    if landmarks is not None:
        working = _rotate_by_eyes(frame, landmarks)

    face = working[y1:y2, x1:x2]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)


def _rotate_by_eyes(frame: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
    left = landmarks.left_eye
    right = landmarks.right_eye
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * scale)
    dy = int(bh * scale)
    return max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)

