from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EMOTION_BIAS, EMOTION_CLASSES, IMAGE_SIZE, MODEL_NUM_CLASSES


class EmotionCNNFactory:
    @staticmethod
    def build(num_classes: int = MODEL_NUM_CLASSES):
        import torch.nn as nn

        return nn.Sequential(
            _conv_block(nn, 1, 64, dropout=0.05),
            _conv_block(nn, 64, 128, dropout=0.10),
            _conv_block(nn, 128, 256, dropout=0.15),
            _conv_block(nn, 256, 256, dropout=0.20),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )


class EmotionRecognizer:
    def __init__(self, weights_path: Path, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for emotion recognition") from exc

        self.torch = torch
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = EmotionCNNFactory.build(num_classes=MODEL_NUM_CLASSES).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, face_gray: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(face_gray)
        index = int(np.argmax(probs))
        return EMOTION_CLASSES[index], float(probs[index])

    def predict_proba(self, face_gray: np.ndarray) -> np.ndarray:
        tensor = self._to_tensor(face_gray)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probs = self.torch.softmax(logits, dim=1)[0]
        probs = probs.detach().cpu().numpy().astype("float32")
        # 模型输出 7 类 [angry, disgust, fear, happy, neutral, sad, surprise]
        # 移除 disgust（索引 1）并重新归一化，映射到 6 类
        probs = np.delete(probs, 1)
        # 定向增强 angry(0)、fear(1)、sad(4) 的概率
        for i, emotion in enumerate(EMOTION_CLASSES):
            boost = EMOTION_BIAS.get(emotion, 0.0)
            if boost > 0.0:
                probs[i] += boost
        clipped = np.clip(probs, 0.0, None)
        total = float(clipped.sum())
        if total <= 1e-8:
            return np.zeros(len(EMOTION_CLASSES), dtype="float32")
        return (clipped / total).astype("float32")

    def _to_tensor(self, face_gray: np.ndarray):
        face = face_gray.astype("float32") / 255.0
        # Match training-time normalization (see train/dataset.py):
        # transforms.ToTensor() -> [0, 1], then Normalize(mean=0.5, std=0.5) -> [-1, 1].
        face = (face - 0.5) / 0.5
        if face.shape != (IMAGE_SIZE, IMAGE_SIZE):
            import cv2

            face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = self.torch.from_numpy(face).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


class NullEmotionRecognizer:
    def predict(self, face_gray: np.ndarray) -> tuple[str, float]:
        return "unknown", 0.0

    def predict_proba(self, face_gray: np.ndarray) -> np.ndarray:
        return np.zeros(len(EMOTION_CLASSES), dtype="float32")


def create_emotion_recognizer(weights_path: Path, device: str | None = None):
    if not weights_path.exists():
        return NullEmotionRecognizer()
    try:
        return EmotionRecognizer(weights_path, device=device)
    except RuntimeError:
        return NullEmotionRecognizer()


def _conv_block(nn, in_channels: int, out_channels: int, dropout: float):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )
