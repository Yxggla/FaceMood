from __future__ import annotations

import cv2
import numpy as np

from .config import EMOTION_CLASSES
from .predictor import FacePrediction
from .stats import emotion_distribution

COLORS = {
    "angry": (40, 40, 230),
    "disgust": (60, 150, 60),
    "fear": (180, 80, 180),
    "happy": (40, 190, 255),
    "neutral": (220, 220, 220),
    "sad": (220, 120, 40),
    "surprise": (30, 220, 220),
    "unknown": (160, 160, 160),
}


def draw_predictions(frame: np.ndarray, predictions: list[FacePrediction]) -> np.ndarray:
    canvas = frame.copy()
    for prediction in predictions:
        color = COLORS.get(prediction.emotion, COLORS["unknown"])
        x1, y1, x2, y2 = prediction.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction.emotion} {prediction.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if prediction.landmarks:
            for point in prediction.landmarks.as_dict().values():
                cv2.circle(canvas, point, 3, color, -1)
    _draw_distribution(canvas, [p.emotion for p in predictions])
    return canvas


def _draw_distribution(frame: np.ndarray, emotions: list[str]) -> None:
    counts = emotion_distribution([e for e in emotions if e in EMOTION_CLASSES])
    x, y = 18, 28
    cv2.putText(frame, "Emotion distribution", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for i, emotion in enumerate(EMOTION_CLASSES):
        text = f"{emotion}: {counts[emotion]}"
        color = COLORS[emotion]
        cv2.putText(frame, text, (x, y + 28 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

