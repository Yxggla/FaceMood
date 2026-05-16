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

from facemood.config import IMAGE_DATA_DIR
from train.dataset import EMOTION_CLASSES, load_split
from train.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FaceMood emotion checkpoint.")
    parser.add_argument("--data-dir", default=str(IMAGE_DATA_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default=str(ROOT / "models" / "exported" / "emotion_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只打印一行 JSON（accuracy、macro_f1），便于对比多个权重。",
    )
    return parser.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    checkpoint = _load_checkpoint(args.weights, device)
    arch = str(checkpoint.get("arch", "cnn")).lower()
    img_size = int(checkpoint.get("img_size", 48))
    preset = str(checkpoint.get("normalize", "fer")).lower()

    dataset = load_split(Path(args.data_dir), args.split, train=False, image_size=img_size, preset=preset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = len(checkpoint["classes"]) if isinstance(checkpoint.get("classes"), list) else len(EMOTION_CLASSES)
    model = build_model(num_classes=num_classes, arch=arch, pretrained=False).to(device)
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
    macro = report.get("macro avg", {})
    summary = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(macro.get("f1-score", 0.0)),
        "arch": arch,
        "split": args.split,
    }
    if args.summary_only:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


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
