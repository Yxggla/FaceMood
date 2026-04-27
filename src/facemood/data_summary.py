from __future__ import annotations

from pathlib import Path

from .config import EMOTION_CLASSES, IMAGE_DATA_DIR


def count_image_dataset(data_dir: Path = IMAGE_DATA_DIR) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        summary[split] = {}
        for emotion in EMOTION_CLASSES:
            emotion_dir = split_dir / emotion
            summary[split][emotion] = len(list(emotion_dir.glob("*.png"))) if emotion_dir.exists() else 0
    return summary


def totals_by_split(summary: dict[str, dict[str, int]]) -> dict[str, int]:
    return {split: sum(counts.values()) for split, counts in summary.items()}


def totals_by_emotion(summary: dict[str, dict[str, int]]) -> dict[str, int]:
    return {
        emotion: sum(summary.get(split, {}).get(emotion, 0) for split in summary)
        for emotion in EMOTION_CLASSES
    }

