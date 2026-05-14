from __future__ import annotations

import numpy as np

from .landmarks import FaceLandmarks


def classify_by_geometry(landmarks: FaceLandmarks) -> tuple[str, float] | None:
    if not landmarks.has_geo():
        return None

    eye_dist = float(np.linalg.norm(
        np.array(landmarks.left_eye, dtype="float32") - np.array(landmarks.right_eye, dtype="float32")
    ))
    if eye_dist < 1.0:
        return None

    def _eyebrow_raise() -> float:
        le = np.array(landmarks.left_eye, dtype="float32")
        re = np.array(landmarks.right_eye, dtype="float32")
        li = np.array(landmarks.left_eyebrow_inner, dtype="float32")
        ri = np.array(landmarks.right_eyebrow_inner, dtype="float32")
        return float(((le[1] - li[1]) + (re[1] - ri[1])) / (2.0 * eye_dist))

    def _eye_openness() -> float:
        lo = float((landmarks.left_eye_bottom[1] - landmarks.left_eye_top[1]) / eye_dist)
        ro = float((landmarks.right_eye_bottom[1] - landmarks.right_eye_top[1]) / eye_dist)
        return (lo + ro) / 2.0

    def _mouth_openness() -> float:
        return float(max(0.0, (landmarks.lower_lip[1] - landmarks.upper_lip[1]) / eye_dist))

    def _mouth_corner_drop() -> float:
        center = (landmarks.upper_lip[1] + landmarks.lower_lip[1]) / 2.0
        left = (landmarks.mouth_left[1] - center) / eye_dist
        right = (landmarks.mouth_right[1] - center) / eye_dist
        return float((left + right) / 2.0)

    def _brow_to_nose() -> float:
        ni = np.array(landmarks.nose_bridge, dtype="float32")
        li = np.array(landmarks.left_eyebrow_inner, dtype="float32")
        ri = np.array(landmarks.right_eyebrow_inner, dtype="float32")
        ld = float(np.linalg.norm(li - ni))
        rd = float(np.linalg.norm(ri - ni))
        return (ld + rd) / (2.0 * eye_dist)

    def _inner_brow_dist() -> float:
        li = np.array(landmarks.left_eyebrow_inner, dtype="float32")
        ri = np.array(landmarks.right_eyebrow_inner, dtype="float32")
        return float(np.linalg.norm(li - ri) / eye_dist)

    brow_raise = _eyebrow_raise()
    eye_open = _eye_openness()
    mouth_open = _mouth_openness()
    corner_drop = _mouth_corner_drop()
    brow_to_nose = _brow_to_nose()
    inner_brow_dist = _inner_brow_dist()

    # --- 1. Fear: 眼睛睁大 + 眉毛抬高 + 嘴巴微张 ---
    if eye_open > 0.11 and brow_raise > 0.10 and mouth_open > 0.02:
        score = min(1.0, 0.50 + (eye_open - 0.10) * 3.0 + (brow_raise - 0.08) * 2.0)
        return "fear", round(score, 2)

    if eye_open > 0.13 and brow_raise > 0.06:
        score = min(1.0, 0.50 + (eye_open - 0.12) * 4.0)
        return "fear", round(score, 2)

    # --- 2. Sad: 眉毛内角上抬 (八字眉) + 嘴角下拉 ---
    if brow_raise > 0.09 and corner_drop > 0.01:
        score = min(1.0, 0.45 + (brow_raise - 0.08) * 3.0 + corner_drop * 5.0)
        return "sad", round(score, 2)

    if brow_raise > 0.12:
        score = min(1.0, 0.45 + (brow_raise - 0.11) * 3.0)
        return "sad", round(score, 2)

    # --- 3. Angry: 眉毛压低 + 眉间靠近 (皱眉) ---
    if brow_raise < 0.06 and inner_brow_dist < 0.60:
        score = min(1.0, 0.50 + (0.08 - brow_raise) * 3.0 + (0.60 - inner_brow_dist) * 1.0)
        return "angry", round(score, 2)

    if brow_raise < 0.01:
        score = min(1.0, 0.55 + (0.03 - brow_raise) * 4.0)
        return "angry", round(score, 2)

    if brow_to_nose < 0.32 and inner_brow_dist < 0.55:
        score = min(1.0, 0.50 + (0.32 - brow_to_nose) * 2.0)
        return "angry", round(score, 2)

    if brow_raise < 0.08 and mouth_open < 0.01:
        return "angry", 0.55

    return None
