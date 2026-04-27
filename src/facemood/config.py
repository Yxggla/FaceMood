from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DATA_DIR = DATA_DIR / "fer2013_7cls_images"
MODEL_PATH = PROJECT_ROOT / "models" / "exported" / "emotion_cnn.pt"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 48
CAMERA_INDEX = 0

