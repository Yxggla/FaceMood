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


def draw_predictions(
    frame: np.ndarray,
    predictions: list[FacePrediction],
    fps: float | None = None,
    recording: bool = False,
) -> np.ndarray:
    canvas = frame.copy()
    for prediction in predictions:
        color = COLORS.get(prediction.emotion, COLORS["unknown"])
        x1, y1, x2, y2 = prediction.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction.emotion} {prediction.confidence:.2f}"
        _draw_label(canvas, label, (x1, max(24, y1 - 8)), color)
        if prediction.landmarks:
            for point in prediction.landmarks.as_dict().values():
                cv2.circle(canvas, point, 3, color, -1)
    if not predictions:
        _draw_label(canvas, "No face detected", (18, 36), COLORS["unknown"])
    _draw_distribution(canvas, [p.emotion for p in predictions])
    _draw_status(canvas, fps=fps, recording=recording)
    return canvas


def _draw_distribution(frame: np.ndarray, emotions: list[str]) -> None:
    counts = emotion_distribution([e for e in emotions if e in EMOTION_CLASSES])
    active = [(emotion, count) for emotion, count in counts.items() if count > 0]
    if not active:
        active = [("all", 0)]
    x, y = 18, frame.shape[0] - 34 - 24 * len(active)
    _draw_label(frame, "Emotion distribution", (x, y), (245, 245, 245))
    for i, (emotion, count) in enumerate(active):
        text = f"{emotion}: {count}"
        color = COLORS.get(emotion, COLORS["unknown"])
        _draw_label(frame, text, (x, y + 26 * (i + 1)), color)


def _draw_status(frame: np.ndarray, fps: float | None, recording: bool) -> None:
    parts = []
    if fps is not None:
        parts.append(f"FPS {fps:.1f}")
    parts.append("S screenshot")
    parts.append("R record")
    parts.append("Q quit")
    text = " | ".join(parts)
    x = 18
    y = frame.shape[0] - 10
    _draw_label(frame, text, (x, y), (245, 245, 245), scale=0.52, thickness=1)
    if recording:
        _draw_label(frame, "REC", (frame.shape[1] - 82, 34), (40, 40, 230), scale=0.7, thickness=2)
        cv2.circle(frame, (frame.shape[1] - 96, 29), 7, (40, 40, 230), -1)


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    x2 = min(frame.shape[1] - 1, x + width + pad * 2)
    y1 = max(0, y - height - baseline - pad)
    y2 = min(frame.shape[0] - 1, y + baseline + pad)
    cv2.rectangle(frame, (x, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, 1)
    cv2.putText(frame, text, (x + pad, y), font, scale, color, thickness, cv2.LINE_AA)
