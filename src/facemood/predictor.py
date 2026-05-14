from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import crop_aligned_face
from .config import (
    BBOX_EMA_ALPHA,
    EMOTION_CLASSES,
    EMOTION_EMA_ALPHA,
    EMOTION_SWITCH_HOLD_FRAMES,
    EMOTION_SWITCH_MARGIN,
    FACE_LOST_TOLERANCE,
    FACE_DETECTOR,
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_SIZE,
    MIN_EMOTION_CONF,
    NO_LANDMARKS_POLICY,
    NO_LANDMARKS_MIN_EMOTION_CONF,
    MODEL_PATH,
    STABILIZE_SINGLE_FACE,
)
from .emotion_model import create_emotion_recognizer
from .face_detector import FaceDetection, OpenCVFaceDetector, create_face_detector
from .geo_rules import classify_by_geometry
from .landmarks import FaceLandmarks, create_landmark_detector


@dataclass(frozen=True)
class FacePrediction:
    bbox: tuple[int, int, int, int]
    landmarks: FaceLandmarks | None
    emotion: str
    confidence: float
    probs: dict[str, float] | None = None

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "landmarks": self.landmarks.as_dict() if self.landmarks else {},
            "emotion": self.emotion,
            "confidence": self.confidence,
        }


class FaceMoodPredictor:
    def __init__(
        self,
        model_path=MODEL_PATH,
        device: str | None = None,
        stable: bool = STABILIZE_SINGLE_FACE,
        detector: str = FACE_DETECTOR,
        min_detection_confidence: float = FACE_MIN_DETECTION_CONFIDENCE,
        min_face_size: int = FACE_MIN_SIZE,
        no_landmarks_policy: str = NO_LANDMARKS_POLICY,
        bbox_alpha: float = BBOX_EMA_ALPHA,
        emotion_alpha: float = EMOTION_EMA_ALPHA,
        switch_margin: float = EMOTION_SWITCH_MARGIN,
        switch_hold_frames: int = EMOTION_SWITCH_HOLD_FRAMES,
        face_lost_tolerance: int = FACE_LOST_TOLERANCE,
        min_emotion_conf: float = MIN_EMOTION_CONF,
        face_detector=None,
        landmark_detector=None,
        emotion_recognizer=None,
    ) -> None:
        self.face_detector = (
            face_detector
            if face_detector is not None
            else create_face_detector(
                detector=detector,
                min_detection_confidence=min_detection_confidence,
                min_face_size=min_face_size,
            )
        )
        self.landmark_detector = landmark_detector if landmark_detector is not None else create_landmark_detector()
        self.emotion_recognizer = (
            emotion_recognizer
            if emotion_recognizer is not None
            else create_emotion_recognizer(Path(model_path), device=device)
        )
        self.stabilizer = TemporalStabilizer(
            enabled=stable,
            bbox_alpha=bbox_alpha,
            emotion_alpha=emotion_alpha,
            switch_margin=switch_margin,
            switch_hold_frames=switch_hold_frames,
            face_lost_tolerance=face_lost_tolerance,
            min_emotion_conf=min_emotion_conf,
        )
        policy = (no_landmarks_policy or "unknown").lower().strip()
        self.no_landmarks_policy = policy if policy in ("discard", "unknown", "allow", "infer") else "infer"
        self.no_landmarks_min_emotion_conf = float(NO_LANDMARKS_MIN_EMOTION_CONF)

    def predict_frame(self, frame: np.ndarray) -> list[FacePrediction]:
        detections = self.face_detector.detect(frame)
        detection = self.stabilizer.select_face(detections, frame.shape[1], frame.shape[0])
        if detection is None:
            ghost = self.stabilizer.on_missed_detection()
            return [ghost] if ghost is not None else []

        landmarks = self.landmark_detector.detect(frame, detection.bbox)
        if landmarks is None and self.no_landmarks_policy == "discard":
            ghost = self.stabilizer.on_missed_detection()
            return [ghost] if ghost is not None else []
        if landmarks is None and self.no_landmarks_policy == "unknown":
            prediction = self.stabilizer.on_detection(
                detection.bbox,
                None,
                np.zeros(len(EMOTION_CLASSES), dtype="float32"),
            )
            return [prediction]
        if landmarks is None and self.no_landmarks_policy == "infer":
            face = crop_aligned_face(frame, detection.bbox, None)
            if face is None:
                ghost = self.stabilizer.on_missed_detection()
                return [ghost] if ghost is not None else []
            probs = self.emotion_recognizer.predict_proba(face)
            probs = _safe_probs(probs)
            if float(np.max(probs)) < self.no_landmarks_min_emotion_conf:
                probs = np.zeros(len(EMOTION_CLASSES), dtype="float32")
            prediction = self.stabilizer.on_detection(detection.bbox, None, probs)
            return [prediction]
        face = crop_aligned_face(frame, detection.bbox, landmarks)
        probs = (
            self.emotion_recognizer.predict_proba(face)
            if face is not None
            else np.zeros(len(EMOTION_CLASSES), dtype="float32")
        )
        probs = _geo_override(probs, landmarks)
        prediction = self.stabilizer.on_detection(detection.bbox, landmarks, probs)
        return [prediction]


class TemporalStabilizer:
    def __init__(
        self,
        enabled: bool,
        bbox_alpha: float,
        emotion_alpha: float,
        switch_margin: float,
        switch_hold_frames: int,
        face_lost_tolerance: int,
        min_emotion_conf: float,
    ) -> None:
        self.enabled = enabled
        self.bbox_alpha = float(np.clip(bbox_alpha, 0.0, 1.0))
        self.emotion_alpha = float(np.clip(emotion_alpha, 0.0, 1.0))
        self.switch_margin = max(0.0, switch_margin)
        self.switch_hold_frames = max(1, switch_hold_frames)
        self.face_lost_tolerance = max(0, face_lost_tolerance)
        self.min_emotion_conf = max(0.0, float(min_emotion_conf))
        self.prev_bbox: tuple[int, int, int, int] | None = None
        self.prev_probs: np.ndarray | None = None
        self.display_emotion = "unknown"
        self.pending_emotion: str | None = None
        self.pending_count = 0
        self.lost_count = 0

    def select_face(self, detections: list[FaceDetection], frame_width: int, frame_height: int) -> FaceDetection | None:
        if not detections:
            return None
        if not self.enabled or len(detections) == 1:
            return detections[0]
        if self.prev_bbox is not None:
            scored = sorted(
                ((_iou(self.prev_bbox, det.bbox), det) for det in detections),
                key=lambda item: item[0],
                reverse=True,
            )
            if scored[0][0] >= 0.15:
                return scored[0][1]
        cx = frame_width * 0.5
        cy = frame_height * 0.5
        return max(detections, key=lambda det: _face_priority(det.bbox, cx, cy))

    def on_detection(
        self,
        bbox: tuple[int, int, int, int],
        landmarks: FaceLandmarks | None,
        probs: np.ndarray,
    ) -> FacePrediction:
        self.lost_count = 0
        if self.prev_bbox is None or not self.enabled:
            smoothed_bbox = bbox
        else:
            smoothed_bbox = _ema_bbox(self.prev_bbox, bbox, self.bbox_alpha)
        self.prev_bbox = smoothed_bbox

        probs = _safe_probs(probs)
        if self.prev_probs is None or not self.enabled:
            smoothed_probs = probs
        else:
            smoothed_probs = self.emotion_alpha * probs + (1.0 - self.emotion_alpha) * self.prev_probs
            smoothed_probs = _safe_probs(smoothed_probs)
        self.prev_probs = smoothed_probs

        emotion, confidence = self._update_emotion_label(smoothed_probs)
        probs_dict = {EMOTION_CLASSES[i]: float(smoothed_probs[i]) for i in range(len(EMOTION_CLASSES))}
        return FacePrediction(smoothed_bbox, landmarks, emotion, confidence, probs=probs_dict)

    def on_missed_detection(self) -> FacePrediction | None:
        if self.prev_bbox is None:
            return None
        self.lost_count += 1
        if self.lost_count > self.face_lost_tolerance:
            self._reset()
            return None
        confidence = max(0.0, 0.5 * (1.0 - self.lost_count / max(1, self.face_lost_tolerance + 1)))
        return FacePrediction(self.prev_bbox, None, self.display_emotion, confidence)

    def _update_emotion_label(self, probs: np.ndarray) -> tuple[str, float]:
        best_index = int(np.argmax(probs))
        best_emotion = EMOTION_CLASSES[best_index]
        best_prob = float(probs[best_index])

        if best_prob < self.min_emotion_conf:
            self.display_emotion = "unknown"
            self.pending_emotion = None
            self.pending_count = 0
            return self.display_emotion, best_prob

        if self.display_emotion == "unknown":
            self.display_emotion = best_emotion
            self.pending_emotion = None
            self.pending_count = 0
            return self.display_emotion, best_prob

        current_index = EMOTION_CLASSES.index(self.display_emotion)
        current_prob = float(probs[current_index])
        margin = best_prob - current_prob

        if best_emotion == self.display_emotion or margin < self.switch_margin:
            self.pending_emotion = None
            self.pending_count = 0
            return self.display_emotion, current_prob

        if self.pending_emotion != best_emotion:
            self.pending_emotion = best_emotion
            self.pending_count = 1
        else:
            self.pending_count += 1

        if self.pending_count >= self.switch_hold_frames:
            self.display_emotion = best_emotion
            self.pending_emotion = None
            self.pending_count = 0
            return self.display_emotion, best_prob
        return self.display_emotion, current_prob

    def _reset(self) -> None:
        self.prev_bbox = None
        self.prev_probs = None
        self.display_emotion = "unknown"
        self.pending_emotion = None
        self.pending_count = 0
        self.lost_count = 0


def _geo_override(probs: np.ndarray, landmarks: FaceLandmarks | None) -> np.ndarray:
    if landmarks is None or not landmarks.has_geo():
        return probs
    result = classify_by_geometry(landmarks)
    if result is None:
        return probs
    geo_emotion, geo_conf = result
    if geo_emotion not in EMOTION_CLASSES:
        return probs

    top_idx = int(np.argmax(probs))
    top_emotion = EMOTION_CLASSES[top_idx]
    top_conf = float(probs[top_idx])
    geo_idx = EMOTION_CLASSES.index(geo_emotion)
    model_geo_conf = float(probs[geo_idx])

    # Geo 强烈判定 angry 时直接接管（用户反馈 angry 最难识别）
    if geo_emotion == "angry" and geo_conf > 0.45:
        new_probs = np.zeros(len(EMOTION_CLASSES), dtype="float32")
        new_probs[geo_idx] = geo_conf
        return _safe_probs(new_probs)

    # Geo 判定 sad/fear/angry 且模型不自信 (< 0.50) 时覆盖
    if top_emotion != geo_emotion and top_conf < 0.50:
        new_probs = np.zeros(len(EMOTION_CLASSES), dtype="float32")
        new_probs[geo_idx] = geo_conf
        return _safe_probs(new_probs)

    # 模型和 geo 判定一致但 geo 置信度更高时提升
    if top_emotion == geo_emotion and model_geo_conf < geo_conf:
        new_probs = np.zeros(len(EMOTION_CLASSES), dtype="float32")
        new_probs[geo_idx] = geo_conf
        return _safe_probs(new_probs)

    return probs


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype("float32"), 0.0, None)
    total = float(clipped.sum())
    if total <= 1e-8:
        return np.zeros(len(EMOTION_CLASSES), dtype="float32")
    return clipped / total


def _ema_bbox(
    prev_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    prev = np.array(prev_bbox, dtype="float32")
    curr = np.array(current_bbox, dtype="float32")
    mixed = alpha * curr + (1.0 - alpha) * prev
    return tuple(int(round(v)) for v in mixed)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 1e-6 else 0.0


def _face_priority(bbox: tuple[int, int, int, int], center_x: float, center_y: float) -> float:
    x1, y1, x2, y2 = bbox
    area = float(max(0, x2 - x1) * max(0, y2 - y1))
    face_cx = (x1 + x2) * 0.5
    face_cy = (y1 + y2) * 0.5
    dist = ((face_cx - center_x) ** 2 + (face_cy - center_y) ** 2) ** 0.5
    return area - dist * 12.0
