from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facemood.face_detector import FaceDetection
from facemood.predictor import FaceMoodPredictor


class _StubFaceDetector:
    def __init__(self, detections: list[FaceDetection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        return list(self._detections)


class _StubLandmarkDetector:
    def __init__(self, landmarks) -> None:
        self._landmarks = landmarks

    def detect(self, frame: np.ndarray, bbox):
        return self._landmarks


class _StubEmotionRecognizer:
    def __init__(self, probs: np.ndarray) -> None:
        self._probs = probs.astype("float32")

    def predict_proba(self, face: np.ndarray) -> np.ndarray:
        return self._probs.copy()


def _probs(emotion_index: int, peak: float = 0.9) -> np.ndarray:
    vec = np.full(7, (1.0 - peak) / 6.0, dtype="float32")
    vec[emotion_index] = peak
    return vec


def test_landmarks_missing_is_treated_as_missed_detection_and_does_not_create_new_prediction():
    frame = np.zeros((240, 320, 3), dtype="uint8")
    detection = FaceDetection((60, 40, 180, 160), 0.99)

    predictor = FaceMoodPredictor(
        stable=True,
        face_lost_tolerance=2,
        no_landmarks_policy="discard",
        face_detector=_StubFaceDetector([detection]),
        landmark_detector=_StubLandmarkDetector(None),
        emotion_recognizer=_StubEmotionRecognizer(_probs(3, 0.9)),
    )

    preds = predictor.predict_frame(frame)
    assert preds == []


def test_landmarks_missing_infer_runs_emotion_model_and_returns_prediction():
    frame = np.zeros((240, 320, 3), dtype="uint8")
    detection = FaceDetection((60, 40, 180, 160), 0.99)

    predictor = FaceMoodPredictor(
        stable=True,
        face_lost_tolerance=2,
        no_landmarks_policy="infer",
        face_detector=_StubFaceDetector([detection]),
        landmark_detector=_StubLandmarkDetector(None),
        emotion_recognizer=_StubEmotionRecognizer(_probs(3, 0.9)),
    )

    preds = predictor.predict_frame(frame)
    assert len(preds) == 1
    assert preds[0].emotion in ("happy", "unknown")


def test_landmarks_missing_after_valid_detection_returns_ghost_and_keeps_last_emotion():
    frame = np.zeros((240, 320, 3), dtype="uint8")
    detection = FaceDetection((60, 40, 180, 160), 0.99)

    fake_landmarks = type(
        "_Lm",
        (object,),
        {
            "left_eye": (0, 0),
            "right_eye": (1, 0),
            "nose": (0, 1),
            "mouth_left": (0, 2),
            "mouth_right": (1, 2),
            "as_dict": lambda self: {
                "left_eye": (0, 0),
                "right_eye": (1, 0),
                "nose": (0, 1),
                "mouth_left": (0, 2),
                "mouth_right": (1, 2),
            },
        },
    )()

    predictor = FaceMoodPredictor(
        stable=True,
        face_lost_tolerance=2,
        face_detector=_StubFaceDetector([detection]),
        landmark_detector=_StubLandmarkDetector(fake_landmarks),
        emotion_recognizer=_StubEmotionRecognizer(_probs(4, 0.9)),
    )
    first = predictor.predict_frame(frame)
    assert len(first) == 1
    assert first[0].emotion == "neutral"

    predictor.landmark_detector = _StubLandmarkDetector(None)
    ghost = predictor.predict_frame(frame)
    assert len(ghost) == 1
    assert ghost[0].emotion == "neutral"
