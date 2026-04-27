from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
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
    parser = argparse.ArgumentParser(description="Train FaceMood 7-class emotion CNN.")
    parser.add_argument("--data-dir", default=str(IMAGE_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "models" / "checkpoints"))
    parser.add_argument("--export-path", default=str(ROOT / "models" / "exported" / "emotion_cnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    train_dataset = load_split(data_dir, "train", train=True, limit=args.limit_train)
    val_dataset = load_split(data_dir, "val", train=False, limit=args.limit_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(EMOTION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_dataset, device) if args.class_weights else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    history = []
    best_acc = -1.0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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
            "classes": EMOTION_CLASSES,
            "history": history,
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, export_path)

    metrics_path = ROOT / "results" / "metrics" / "training_history.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_epoch(model, loader, criterion, optimizer, device):
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


def _class_weights(dataset, device):
    base_dataset = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", range(len(base_dataset)))
    counts = torch.zeros(len(EMOTION_CLASSES), dtype=torch.float32)
    for index in indices:
        counts[int(base_dataset.targets[index])] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (len(EMOTION_CLASSES) * counts)
    return weights.to(device)


if __name__ == "__main__":
    main()
