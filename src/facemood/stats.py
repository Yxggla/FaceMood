from __future__ import annotations

from collections import Counter

from .config import EMOTION_CLASSES


def emotion_distribution(emotions: list[str]) -> dict[str, int]:
    counts = Counter(emotions)
    return {emotion: counts.get(emotion, 0) for emotion in EMOTION_CLASSES}

