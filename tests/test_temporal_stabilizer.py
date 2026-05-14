from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facemood.face_detector import FaceDetection
from facemood.predictor import TemporalStabilizer


def _stabilizer() -> TemporalStabilizer:
    return TemporalStabilizer(
        enabled=True,
        bbox_alpha=0.5,
        emotion_alpha=1.0,
        switch_margin=0.1,
        switch_hold_frames=2,
        face_lost_tolerance=2,
        min_emotion_conf=0.0,
    )


def _probs(emotion_index: int, peak: float = 0.8) -> np.ndarray:
    vec = np.full(7, (1.0 - peak) / 6.0, dtype="float32")
    vec[emotion_index] = peak
    return vec


def test_select_face_prefers_high_iou_with_previous_bbox():
    stabilizer = _stabilizer()
    stabilizer.prev_bbox = (100, 100, 200, 200)
    detections = [
        FaceDetection((105, 104, 205, 204), 1.0),
        FaceDetection((300, 300, 420, 420), 1.0),
    ]
    selected = stabilizer.select_face(detections, frame_width=640, frame_height=480)
    assert selected is not None
    assert selected.bbox == (105, 104, 205, 204)


def test_bbox_ema_applies_smoothing():
    stabilizer = _stabilizer()
    stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.9))
    prediction = stabilizer.on_detection((120, 80, 220, 180), None, _probs(3, 0.9))
    assert prediction.bbox == (110, 90, 210, 190)


def test_missed_detection_uses_tolerance_then_clears():
    stabilizer = _stabilizer()
    stabilizer.on_detection((100, 100, 200, 200), None, _probs(4, 0.8))
    ghost1 = stabilizer.on_missed_detection()
    ghost2 = stabilizer.on_missed_detection()
    ghost3 = stabilizer.on_missed_detection()
    assert ghost1 is not None
    assert ghost2 is not None
    assert ghost3 is None


def test_emotion_switch_requires_margin_and_hold_frames():
    stabilizer = _stabilizer()
    first = stabilizer.on_detection((100, 100, 200, 200), None, _probs(4, 0.75))
    assert first.emotion == "neutral"

    # Candidate class rises, but first frame should not switch yet.
    second = stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.85))
    assert second.emotion == "neutral"

    # Second consecutive frame with clear margin should switch.
    third = stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.85))
    assert third.emotion == "happy"


def test_low_confidence_predictions_are_reported_as_unknown():
    stabilizer = TemporalStabilizer(
        enabled=True,
        bbox_alpha=0.5,
        emotion_alpha=1.0,
        switch_margin=0.1,
        switch_hold_frames=2,
        face_lost_tolerance=2,
        min_emotion_conf=0.6,
    )
    pred = stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.55))
    assert pred.emotion == "unknown"


def test_unknown_recovers_to_emotion_when_confidence_exceeds_threshold():
    stabilizer = TemporalStabilizer(
        enabled=True,
        bbox_alpha=0.5,
        emotion_alpha=1.0,
        switch_margin=0.1,
        switch_hold_frames=2,
        face_lost_tolerance=2,
        min_emotion_conf=0.6,
    )
    low = stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.55))
    assert low.emotion == "unknown"

    high = stabilizer.on_detection((100, 100, 200, 200), None, _probs(3, 0.85))
    assert high.emotion == "happy"
