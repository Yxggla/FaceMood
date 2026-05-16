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
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


WORKDIR = Path("/kaggle/working")
DATA_ROOT = WORKDIR / "fer2013_7cls_images"
CHECKPOINT_PATH = WORKDIR / "emotion_cnn_v2.pt"
HISTORY_PATH = WORKDIR / "training_history_cnn_v2.json"
REPORT_PATH = WORKDIR / "test_classification_report_cnn_v2.json"

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

IMAGE_SIZE = 48
BATCH_SIZE = 256
EPOCHS = 40
LR = 7e-4
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.08
MIXUP_ALPHA = 0.20
GRAD_CLIP = 1.0


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
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.90, 1.10)),
            ]
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    if train:
        ops.append(transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0.0))
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


class EmotionCNNV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.stage1 = self._stage(48, 64, 0.05)
        self.stage2 = self._stage(64, 128, 0.08)
        self.stage3 = self._stage(128, 192, 0.12)
        self.stage4 = self._stage(192, 256, 0.16)
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

    def _stage(self, in_channels: int, out_channels: int, dropout: float):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels)
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


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = max(16, channels // 8)
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


def run_epoch(model, loader, criterion, optimizer, device: torch.device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        target_for_acc = labels
        target_b = None
        lam = 1.0
        if training and MIXUP_ALPHA > 1e-8:
            lam = float(torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample(()).item())
            index = torch.randperm(images.size(0), device=images.device)
            images = lam * images + (1.0 - lam) * images[index]
            target_b = labels[index]
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if training and target_b is not None:
            loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, target_b)
        else:
            loss = criterion(logits, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == target_for_acc).sum().item())
        total_samples += target_for_acc.size(0)

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
    return classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)


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

    model = EmotionCNNV2(len(EMOTION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset, device), label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)

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
            "arch": "cnn_v2",
            "img_size": IMAGE_SIZE,
            "normalize": "fer",
            "train_config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "mixup_alpha": MIXUP_ALPHA,
                "grad_clip": GRAD_CLIP,
            },
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
