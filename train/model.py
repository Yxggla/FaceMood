from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facemood.emotion_model import build_torch_emotion_model


def build_model(num_classes: int = 7, arch: str = "cnn", pretrained: bool = True):
    return build_torch_emotion_model(arch, num_classes, pretrained=pretrained)

