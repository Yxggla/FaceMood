from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DATA_DIR = DATA_DIR / "fer2013_7cls_images"
MODEL_PATH = PROJECT_ROOT / "models" / "exported" / "emotion_cnn.pt"

# 用户可见的情绪类别（已移除 disgust）
EMOTION_CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
# 原始模型输出维度（含 disgust 共 7 类，与预训练权重匹配）
MODEL_NUM_CLASSES = 7
IMAGE_SIZE = 48
CAMERA_INDEX = 0

FACE_DETECTOR = "mediapipe"  # mediapipe | haar | auto
FACE_MIN_DETECTION_CONFIDENCE = 0.6
FACE_MIN_SIZE = 60
NO_LANDMARKS_POLICY = "infer"  # discard | unknown | allow | infer
NO_LANDMARKS_MIN_EMOTION_CONF = 0.40
MIN_EMOTION_CONF = 0.15

# 定向增强难识别的情绪（模型输出概率加权，值越大越容易识别）
EMOTION_BIAS = {
    "angry": 0.18,
    "sad": 0.14,
    "fear": 0.12,
}

STABILIZE_SINGLE_FACE = True
BBOX_EMA_ALPHA = 0.40
EMOTION_EMA_ALPHA = 0.85
FACE_LOST_TOLERANCE = 6
EMOTION_SWITCH_MARGIN = 0.005
EMOTION_SWITCH_HOLD_FRAMES = 1

