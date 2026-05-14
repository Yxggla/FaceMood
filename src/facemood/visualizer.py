from __future__ import annotations

import cv2
import numpy as np

from .config import EMOTION_CLASSES
from .predictor import FacePrediction

COLORS = {
    "angry": (40, 40, 230),
    "fear": (180, 80, 180),
    "happy": (40, 190, 255),
    "neutral": (220, 220, 220),
    "sad": (220, 120, 40),
    "surprise": (30, 220, 220),
    "unknown": (160, 160, 160),
}

_BAR_COLORS = {
    "angry": (40, 40, 200),
    "fear": (160, 60, 160),
    "happy": (30, 170, 230),
    "neutral": (180, 180, 180),
    "sad": (200, 100, 30),
    "surprise": (20, 200, 200),
}

MARGIN = 14
BAR_GAP = 2
BAR_HEIGHT = 16
PANEL_W = 170
PANEL_PAD = 10


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
        _draw_label_bg(canvas, label, (x1, max(24, y1 - 12)), color)
        if prediction.landmarks:
            for point in prediction.landmarks.as_dict().values():
                cv2.circle(canvas, point, 3, color, -1)

    if not predictions:
        _draw_label_bg(canvas, "No face detected", (MARGIN, MARGIN + 24), COLORS["unknown"])

    _draw_status_bar(canvas, fps=fps, recording=recording)
    _draw_prob_panel(canvas, predictions)

    return canvas


def _draw_status_bar(frame: np.ndarray, fps: float | None, recording: bool) -> None:
    parts = []
    if fps is not None:
        parts.append(f"FPS {fps:.0f}")
    parts.append("S screenshot")
    parts.append("R record")
    parts.append("Q quit")
    text = " | ".join(parts)
    x = MARGIN
    y = frame.shape[0] - 10
    _draw_label_bg(frame, text, (x, y), (245, 245, 245), scale=0.45, thickness=1)
    if recording:
        _draw_label_bg(frame, "REC", (frame.shape[1] - 76, 34), (40, 40, 230), scale=0.6, thickness=2)
        cv2.circle(frame, (frame.shape[1] - 90, 29), 7, (40, 40, 230), -1)


def _draw_prob_panel(frame: np.ndarray, predictions: list[FacePrediction]) -> None:
    if not predictions:
        return

    pred = predictions[0]
    if pred.probs is None:
        return

    items = sorted(pred.probs.items(), key=lambda x: x[1], reverse=True)
    h = frame.shape[0]
    w = frame.shape[1]

    n = len(items)
    panel_h = 20 + n * (BAR_HEIGHT + BAR_GAP) + 6
    panel_x = w - MARGIN - PANEL_W
    panel_y = h // 2 - panel_h // 2

    overlay = frame[panel_y:panel_y + panel_h, panel_x:panel_x + PANEL_W]
    cv2.rectangle(overlay, (0, 0), (PANEL_W, panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, 0), (PANEL_W, panel_h), (70, 70, 70), 1)

    _put_text(frame, "Probabilities", (panel_x + PANEL_PAD, panel_y + 14), (220, 220, 220), scale=0.45, thickness=1)

    bar_y = panel_y + 22
    bar_max_w = PANEL_W - PANEL_PAD * 2
    for emotion, prob in items:
        pct = prob * 100
        bw = int(bar_max_w * prob)
        bx = panel_x + PANEL_PAD
        color = _BAR_COLORS.get(emotion, (160, 160, 160))
        cv2.rectangle(frame, (bx, bar_y), (bx + bar_max_w, bar_y + BAR_HEIGHT), (45, 45, 45), -1)
        if bw > 0:
            cv2.rectangle(frame, (bx, bar_y), (bx + bw, bar_y + BAR_HEIGHT), color, -1)
        _put_text(frame, f"{emotion} {pct:.0f}%", (bx + 2, bar_y + BAR_HEIGHT - 3), (255, 255, 255), scale=0.38, thickness=1)
        bar_y += BAR_HEIGHT + BAR_GAP


def _draw_label_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
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


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
