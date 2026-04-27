from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facemood.config import EMOTION_CLASSES, IMAGE_DATA_DIR
from train.dataset import load_split
from train.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FaceMood emotion CNN.")
    parser.add_argument("--data-dir", default=str(IMAGE_DATA_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default=str(ROOT / "models" / "exported" / "emotion_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset = load_split(Path(args.data_dir), args.split, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(num_classes=len(EMOTION_CLASSES)).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_CLASSES))))

    metrics_dir = ROOT / "results" / "metrics"
    figures_dir = ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.split}_classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_confusion_matrix(matrix, figures_dir / f"{args.split}_confusion_matrix.png")
    print(json.dumps(report["accuracy"], indent=2))


def plot_confusion_matrix(matrix, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(EMOTION_CLASSES)), EMOTION_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTION_CLASSES)), EMOTION_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FER2013 7-class Confusion Matrix")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, int(matrix[y, x]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
