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

from facemood.config import IMAGE_DATA_DIR
from facemood.emotion_model import IMAGENET_BACKBONE_ARCHS
from train.dataset import EMOTION_CLASSES, load_split
from train.model import build_model

_EMOTION_ARCH_CHOICES = ("cnn", "cnn_v2", "resnet18")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FaceMood 7-class emotion model.",
        epilog=(
            "Kaggle(CUDA) 示例:\n"
            "  python train/train_emotion.py --data-dir /kaggle/input/your-fer --device cuda "
            "--arch cnn_v2 --epochs 40 --batch-size 256 --class-weights "
            "--export-path /kaggle/working/emotion_kaggle.pt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(IMAGE_DATA_DIR))
    parser.add_argument("--arch", default="cnn_v2", choices=list(_EMOTION_ARCH_CHOICES))
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="输入边长（默认 cnn=48，ResNet/EfficientNet 等 ImageNet 骨干=224）。",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNet 骨干: 不加载 ImageNet 预训练权重。",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "models" / "checkpoints"))
    parser.add_argument("--export-path", default=str(ROOT / "models" / "exported" / "emotion_cnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    img_size = args.img_size if args.img_size is not None else (224 if args.arch in IMAGENET_BACKBONE_ARCHS else 48)
    preset = "imagenet" if args.arch in IMAGENET_BACKBONE_ARCHS else "fer"
    pretrained = not args.no_pretrained

    train_dataset = load_split(
        data_dir,
        "train",
        train=True,
        limit=args.limit_train,
        image_size=img_size,
        preset=preset,
    )
    val_dataset = load_split(
        data_dir,
        "val",
        train=False,
        limit=args.limit_val,
        image_size=img_size,
        preset=preset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        num_classes=len(EMOTION_CLASSES),
        arch=args.arch,
        pretrained=pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_dataset, device) if args.class_weights else None,
        label_smoothing=max(0.0, float(args.label_smoothing)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args)
    history = []
    best_acc = -1.0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=args.mixup_alpha,
            grad_clip=args.grad_clip,
        )
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
        if scheduler is not None:
            scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "classes": list(EMOTION_CLASSES),
            "history": history,
            "arch": args.arch,
            "img_size": img_size,
            "normalize": preset,
            "train_args": vars(args),
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, export_path)

    metrics_path = ROOT / "results" / "metrics" / "training_history.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_epoch(model, loader, criterion, optimizer, device, *, mixup_alpha: float = 0.0, grad_clip: float = 0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        target_for_acc = labels
        mixup_lam = 1.0
        mixup_index = None
        if training and mixup_alpha > 1e-8:
            images, labels, target_b, mixup_lam, mixup_index = _mixup_batch(images, labels, mixup_alpha)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if training and mixup_index is not None:
            loss = mixup_lam * criterion(logits, labels) + (1.0 - mixup_lam) * criterion(logits, target_b)
        else:
            loss = criterion(logits, labels)
        if training:
            loss.backward()
            if grad_clip > 1e-8:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == target_for_acc).sum().item())
        total_samples += target_for_acc.size(0)

    return total_loss / total_samples, total_correct / total_samples


def _mixup_batch(images, labels, alpha: float):
    lam = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam, index


def _build_scheduler(optimizer, args):
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)


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
