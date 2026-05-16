from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EMOTION_BIAS, EMOTION_CLASSES, IMAGE_SIZE, MODEL_NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class EmotionCNNV2Factory:
    @staticmethod
    def build(num_classes: int = MODEL_NUM_CLASSES):
        import torch.nn as nn

        return _EmotionCNNV2(nn, num_classes=num_classes)


def build_resnet18(num_classes: int, *, pretrained: bool = True):
    from torch import nn

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("torchvision is required for ResNet18") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ImageNet 预训练骨干：灰度复制成 3 通道，用 train/dataset.py 的 imagenet preset
IMAGENET_BACKBONE_ARCHS = frozenset({"resnet18"})


def build_torch_emotion_model(arch: str, num_classes: int, *, pretrained: bool = True):
    arch = (arch or "cnn").lower()
    if arch == "cnn":
        return EmotionCNNFactory.build(num_classes=num_classes)
    if arch == "cnn_v2":
        return EmotionCNNV2Factory.build(num_classes=num_classes)
    if arch == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown emotion model arch: {arch!r}. Supported: 'cnn', 'cnn_v2', 'resnet18'.")


class EmotionRecognizer:
    def __init__(
        self,
        weights_path: Path,
        device: str | None = None,
        *,
        use_emotion_bias: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for emotion recognition") from exc

        self.torch = torch
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        arch = str(checkpoint.get("arch", "cnn")).lower()
        num_classes = len(checkpoint["classes"]) if isinstance(checkpoint.get("classes"), list) else MODEL_NUM_CLASSES
        self.arch = arch
        self.img_size = int(checkpoint.get("img_size", IMAGE_SIZE))
        self.normalize = str(checkpoint.get("normalize", "fer")).lower()
        self.use_emotion_bias = bool(use_emotion_bias)
        self.model = build_torch_emotion_model(arch, num_classes, pretrained=False).to(self.device)
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
        if self.use_emotion_bias:
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
        import cv2

        face = face_gray.astype("float32")
        if float(face.max()) > 1.5:
            face = face / 255.0
        h, w = self.img_size, self.img_size
        if face.shape != (h, w):
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
        if self.normalize == "imagenet":
            stacked = np.stack([face, face, face], axis=0)
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            stacked = (stacked - mean) / std
            tensor = self.torch.from_numpy(stacked).unsqueeze(0)
        else:
            # Match train/dataset.py fer preset: ToTensor [0,1] then Normalize(0.5, 0.5) -> [-1, 1]
            face = (face - 0.5) / 0.5
            tensor = self.torch.from_numpy(face).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


class NullEmotionRecognizer:
    def predict(self, face_gray: np.ndarray) -> tuple[str, float]:
        return "unknown", 0.0

    def predict_proba(self, face_gray: np.ndarray) -> np.ndarray:
        return np.zeros(len(EMOTION_CLASSES), dtype="float32")


def create_emotion_recognizer(
    weights_path: Path,
    device: str | None = None,
    *,
    use_emotion_bias: bool = True,
):
    if not weights_path.exists():
        print(f"[FaceMood] Emotion model not found: {weights_path}")
        return NullEmotionRecognizer()
    try:
        return EmotionRecognizer(weights_path, device=device, use_emotion_bias=use_emotion_bias)
    except RuntimeError as exc:
        print(f"[FaceMood] Failed to load emotion model '{weights_path}': {exc}")
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


def _residual_stage(nn, in_channels: int, out_channels: int, dropout: float):
    return nn.Sequential(
        _ResidualBlock(nn, in_channels, out_channels),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class _EmotionCNNV2:
    def __new__(cls, nn, num_classes: int):
        class EmotionCNNV2(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(48),
                    nn.SiLU(inplace=True),
                )
                self.stage1 = _residual_stage(nn, 48, 64, dropout=0.05)
                self.stage2 = _residual_stage(nn, 64, 128, dropout=0.08)
                self.stage3 = _residual_stage(nn, 128, 192, dropout=0.12)
                self.stage4 = _residual_stage(nn, 192, 256, dropout=0.16)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.LayerNorm(256),
                    nn.Dropout(0.35),
                    nn.Linear(256, 192),
                    nn.SiLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(192, num_classes),
                )

            def forward(self, x):
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                return self.head(x)

        return EmotionCNNV2()


class _ResidualBlock:
    def __new__(cls, nn, in_channels: int, out_channels: int):
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.act1 = nn.SiLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.se = _SqueezeExcite(nn, out_channels)
                self.shortcut = (
                    nn.Identity()
                    if in_channels == out_channels
                    else nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )
                )
                self.act2 = nn.SiLU(inplace=True)

            def forward(self, x):
                residual = self.shortcut(x)
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.act1(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.se(out)
                out = out + residual
                return self.act2(out)

        return ResidualBlock()


class _SqueezeExcite:
    def __new__(cls, nn, channels: int):
        reduced = max(16, channels // 8)

        class SqueezeExcite(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
                self.act = nn.SiLU(inplace=True)
                self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
                self.gate = nn.Sigmoid()

            def forward(self, x):
                scale = self.pool(x)
                scale = self.fc1(scale)
                scale = self.act(scale)
                scale = self.fc2(scale)
                scale = self.gate(scale)
                return x * scale

        return SqueezeExcite()
