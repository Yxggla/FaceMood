from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import EMOTION_CLASSES
from .predictor import FacePrediction


@dataclass
class AutoScreenshotResult:
    emotion: str
    count: int
    path: Path


@dataclass
class AutoEmotionScreenshotter:
    output_dir: Path
    shots_per_emotion: int = 10
    stable_frames: int = 5
    interval_frames: int = 3
    min_confidence: float = 0.0
    target_emotions: tuple[str, ...] = tuple(EMOTION_CLASSES)
    counts: dict[str, int] = field(init=False)
    current_emotion: str | None = field(default=None, init=False)
    current_streak: int = field(default=0, init=False)
    frame_index: int = field(default=0, init=False)
    last_capture_frame: int = field(default=-10_000, init=False)

    def __post_init__(self) -> None:
        self.shots_per_emotion = max(1, int(self.shots_per_emotion))
        self.stable_frames = max(1, int(self.stable_frames))
        self.interval_frames = max(1, int(self.interval_frames))
        self.min_confidence = max(0.0, float(self.min_confidence))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counts = {emotion: 0 for emotion in self.target_emotions}
        for emotion in self.target_emotions:
            (self.output_dir / emotion).mkdir(parents=True, exist_ok=True)

    @property
    def total_saved(self) -> int:
        return sum(self.counts.values())

    @property
    def total_target(self) -> int:
        return self.shots_per_emotion * len(self.counts)

    @property
    def is_complete(self) -> bool:
        return all(count >= self.shots_per_emotion for count in self.counts.values())

    def update(self, frame: np.ndarray, predictions: list[FacePrediction]) -> list[AutoScreenshotResult]:
        self.frame_index += 1
        if self.is_complete:
            return []

        emotion = self._eligible_emotion(predictions)
        if emotion is None:
            self._reset_streak()
            return []

        if emotion == self.current_emotion:
            self.current_streak += 1
        else:
            self.current_emotion = emotion
            self.current_streak = 1

        if self.current_streak < self.stable_frames:
            return []
        if self.counts[emotion] >= self.shots_per_emotion:
            return []
        if self.frame_index - self.last_capture_frame < self.interval_frames:
            return []

        next_count = self.counts[emotion] + 1
        path = self._path_for(emotion, next_count)
        if not cv2.imwrite(str(path), frame):
            return []

        self.counts[emotion] = next_count
        self.last_capture_frame = self.frame_index
        return [AutoScreenshotResult(emotion=emotion, count=next_count, path=path)]

    def _eligible_emotion(self, predictions: list[FacePrediction]) -> str | None:
        if not predictions:
            return None
        prediction = predictions[0]
        emotion = prediction.emotion
        if emotion not in self.counts:
            return None
        if self.counts[emotion] >= self.shots_per_emotion:
            return None
        if prediction.confidence < self.min_confidence:
            return None
        return emotion

    def _path_for(self, emotion: str, count: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{emotion}_{count:02d}_{timestamp}.png"
        return self.output_dir / emotion / filename

    def _reset_streak(self) -> None:
        self.current_emotion = None
        self.current_streak = 0
