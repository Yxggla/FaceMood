from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

EMOTION_MAP = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}
USAGE_MAP = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FER2013 CSV to ImageFolder layout.")
    parser.add_argument("--csv", default="data/fer2013/fer2013.csv")
    parser.add_argument("--out", default="data/fer2013_7cls_images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    counters: dict[tuple[str, str], int] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = USAGE_MAP[row["Usage"]]
            emotion = EMOTION_MAP[row["emotion"]]
            pixels = [int(value) for value in row["pixels"].split()]
            image = Image.new("L", (48, 48))
            image.putdata(pixels)

            key = (split, emotion)
            index = counters.get(key, 0)
            counters[key] = index + 1
            target_dir = out_dir / split / emotion
            target_dir.mkdir(parents=True, exist_ok=True)
            image.save(target_dir / f"fer_{index:05d}.png")

    print(f"Wrote images to {out_dir}")


if __name__ == "__main__":
    main()

