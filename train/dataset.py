from __future__ import annotations

from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets, transforms

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build_transform(train: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    return transforms.Compose(ops)


def load_split(data_dir: Path, split: str, train: bool, limit: int | None = None):
    dataset = datasets.ImageFolder(data_dir / split, transform=build_transform(train=train))
    if dataset.classes != EMOTION_CLASSES:
        raise ValueError(f"Unexpected class order: {dataset.classes}. Expected: {EMOTION_CLASSES}")
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset

