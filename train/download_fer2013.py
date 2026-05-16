"""从 Hugging Face 拉取 FER2013（与常见 train/publicTest/privateTest 规模一致），导出为 ImageFolder。

无需 Kaggle API。默认输出到 data/fer2013_7cls_images/，与 train/prepare_fer2013.py 结果布局一致。
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

SPLIT_FILES = {
    "train": "data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
    "val": "data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
    "test": "data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
}

HF_REPO = "Aaryan333/fer2013_train_publicTest_privateTest"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FER2013 from Hugging Face as ImageFolder.")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "fer2013_7cls_images")
    p.add_argument("--repo", default=HF_REPO)
    return p.parse_args()


def _image_bytes(cell) -> bytes:
    if isinstance(cell, dict) and "bytes" in cell:
        return cell["bytes"]
    if isinstance(cell, bytes):
        return cell
    raise TypeError(f"Unexpected image cell type: {type(cell)}")


def main() -> None:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image
    except ImportError as e:
        raise SystemExit("请先安装: pip install huggingface_hub pandas pyarrow pillow") from e

    args = parse_args()
    out_dir: Path = args.out

    for split, remote in SPLIT_FILES.items():
        path = hf_hub_download(args.repo, remote, repo_type="dataset")
        df = pd.read_parquet(path)
        if len(df.columns) != 2 or "label" not in df.columns:
            raise SystemExit(f"Unexpected columns in {remote}: {df.columns.tolist()}")

        counters: dict[str, int] = {}
        for _, row in df.iterrows():
            label = int(row["label"])
            if label not in EMOTION_NAMES:
                raise SystemExit(f"Unknown label {label}")
            emotion = EMOTION_NAMES[label]
            raw = _image_bytes(row["image"])
            img = Image.open(io.BytesIO(raw)).convert("L")
            idx = counters.get(emotion, 0)
            counters[emotion] = idx + 1
            target = out_dir / split / emotion
            target.mkdir(parents=True, exist_ok=True)
            img.save(target / f"fer_{idx:05d}.png")

        print(f"{split}: wrote {len(df)} images -> {out_dir / split}")

    print(f"Done. ImageFolder root: {out_dir}")


if __name__ == "__main__":
    main()
