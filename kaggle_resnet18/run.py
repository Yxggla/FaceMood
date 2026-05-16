from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path


def ensure_package(name: str) -> None:
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])


for pkg in ("huggingface_hub", "pandas", "pyarrow", "PIL"):
    ensure_package("pillow" if pkg == "PIL" else pkg)

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


WORKDIR = Path("/kaggle/working")
DATA_ROOT = WORKDIR / "fer2013_7cls_images"
CHECKPOINT_PATH = WORKDIR / "emotion_resnet18.pt"
HISTORY_PATH = WORKDIR / "training_history_resnet18.json"
REPORT_PATH = WORKDIR / "test_classification_report_resnet18.json"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
HF_REPO = "Aaryan333/fer2013_train_publicTest_privateTest"
SPLIT_FILES = {
    "train": "data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
    "val": "data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
    "test": "data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
}
EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_STEP = 5
LR_GAMMA = 0.5


def image_bytes(cell) -> bytes:
    if isinstance(cell, dict) and "bytes" in cell:
        return cell["bytes"]
    if isinstance(cell, bytes):
        return cell
    raise TypeError(f"Unexpected image payload type: {type(cell)}")


def prepare_dataset() -> None:
    if DATA_ROOT.exists():
        print(f"Dataset already exists at {DATA_ROOT}")
        return

    for split, remote in SPLIT_FILES.items():
        path = hf_hub_download(HF_REPO, remote, repo_type="dataset")
        df = pd.read_parquet(path)
        counters: dict[str, int] = {}
        for _, row in df.iterrows():
            label = int(row["label"])
            emotion = EMOTION_NAMES[label]
            raw = image_bytes(row["image"])
            img = Image.open(io.BytesIO(raw)).convert("L")
            idx = counters.get(emotion, 0)
            counters[emotion] = idx + 1
            target = DATA_ROOT / split / emotion
            target.mkdir(parents=True, exist_ok=True)
            img.save(target / f"fer_{idx:05d}.png")
        print(f"{split}: wrote {len(df)} images")


def build_transform(train: bool) -> transforms.Compose:
    ops: list = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            ]
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=list(ResNet18_Weights.DEFAULT.transforms().mean), std=list(ResNet18_Weights.DEFAULT.transforms().std)))
    if train:
        ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0.0))
    return transforms.Compose(ops)


def load_split(split: str, train: bool):
    dataset = datasets.ImageFolder(DATA_ROOT / split, transform=build_transform(train))
    if dataset.classes != EMOTION_CLASSES:
        raise ValueError(f"Unexpected class order: {dataset.classes}")
    return dataset


def class_weights(dataset, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(EMOTION_CLASSES), dtype=torch.float32)
    for target in dataset.targets:
        counts[int(target)] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (len(EMOTION_CLASSES) * counts)
    return weights.to(device)


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTION_CLASSES))
    return model


def run_epoch(model, loader, criterion, optimizer, device: torch.device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device: torch.device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    return report


def main() -> None:
    print("Preparing dataset...")
    prepare_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = load_split("train", train=True)
    val_dataset = load_split("val", train=False)
    test_dataset = load_split("test", train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    history: list[dict] = []
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "classes": list(EMOTION_CLASSES),
            "history": history,
            "arch": "resnet18",
            "img_size": IMAGE_SIZE,
            "normalize": "imagenet",
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, CHECKPOINT_PATH)

    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    report = evaluate(model, test_loader, device)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "best_val_acc": best_val_acc,
        "test_accuracy": float(report["accuracy"]),
        "test_macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "checkpoint": str(CHECKPOINT_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
