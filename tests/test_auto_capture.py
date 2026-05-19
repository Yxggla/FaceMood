from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facemood.auto_capture import AutoEmotionScreenshotter
from facemood.predictor import FacePrediction


def _prediction(emotion: str, confidence: float = 0.8) -> FacePrediction:
    return FacePrediction((10, 10, 80, 80), None, emotion, confidence)


def test_auto_screenshotter_waits_for_stable_emotion(tmp_path: Path):
    screenshotter = AutoEmotionScreenshotter(
        tmp_path,
        shots_per_emotion=2,
        stable_frames=3,
        interval_frames=1,
        target_emotions=("happy", "sad"),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert screenshotter.update(frame, [_prediction("happy")]) == []
    assert screenshotter.update(frame, [_prediction("sad")]) == []
    assert screenshotter.update(frame, [_prediction("happy")]) == []
    assert screenshotter.update(frame, [_prediction("happy")]) == []

    saved = screenshotter.update(frame, [_prediction("happy")])

    assert len(saved) == 1
    assert saved[0].emotion == "happy"
    assert saved[0].count == 1
    assert saved[0].path.exists()


def test_auto_screenshotter_caps_each_emotion_count(tmp_path: Path):
    screenshotter = AutoEmotionScreenshotter(
        tmp_path,
        shots_per_emotion=2,
        stable_frames=1,
        interval_frames=1,
        target_emotions=("happy",),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    saved = []
    for _ in range(5):
        saved.extend(screenshotter.update(frame, [_prediction("happy")]))

    assert len(saved) == 2
    assert screenshotter.counts["happy"] == 2
    assert sorted(path.name for path in (tmp_path / "happy").glob("*.png")) == sorted(
        result.path.name for result in saved
    )
