from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets, transforms

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool, *, image_size: int = 48, preset: str = "fer"):
    preset = preset.lower()
    if preset == "imagenet":
        ops: list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
        ]
    else:
        ops = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
        ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            ]
        )
    ops.append(transforms.ToTensor())
    if preset == "imagenet":
        ops.append(transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)))
    else:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    if train:
        ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0.0))
    return transforms.Compose(ops)


def load_split(
    data_dir: Path,
    split: str,
    train: bool,
    limit: int | None = None,
    *,
    image_size: int = 48,
    preset: str = "fer",
):
    dataset = datasets.ImageFolder(
        data_dir / split,
        transform=build_transform(train=train, image_size=image_size, preset=preset),
    )
    if dataset.classes != EMOTION_CLASSES:
        raise ValueError(f"Unexpected class order: {dataset.classes}. Expected: {EMOTION_CLASSES}")
    if limit is not None:
        dataset = Subset(dataset, _balanced_indices(dataset.targets, limit))
    return dataset


def _balanced_indices(targets: list[int], limit: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    selected: list[int] = []
    class_ids = sorted(by_class)
    offset = 0
    while len(selected) < min(limit, len(targets)):
        added = False
        for class_id in class_ids:
            items = by_class[class_id]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected
