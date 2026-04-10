"""Training entrypoint
"""
"""Training entrypoint for all tasks.

Usage examples:
    python train.py --task classification --data ./data --epochs 30 --lr 1e-3
    python train.py --task localization   --data ./data --epochs 30 --lr 1e-4
    python train.py --task segmentation   --data ./data --epochs 30 --lr 1e-4
"""
"""Training entrypoint for all tasks."""

# import sys
import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
# import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute per-sample IoU for a batch of (cx,cy,w,h) boxes."""
    px1 = pred[:, 0] - pred[:, 2] / 2;  px2 = pred[:, 0] + pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2;  py2 = pred[:, 1] + pred[:, 3] / 2
    tx1 = target[:, 0] - target[:, 2] / 2; tx2 = target[:, 0] + target[:, 2] / 2
    ty1 = target[:, 1] - target[:, 3] / 2; ty2 = target[:, 1] + target[:, 3] / 2

    ix1 = torch.max(px1, tx1); ix2 = torch.min(px2, tx2)
    iy1 = torch.max(py1, ty1); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    return inter / (pa + ta - inter + eps)


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Macro-average Dice score over all classes."""
    pred = pred_logits.argmax(dim=1)
    num_classes = pred_logits.shape[1]
    scores = []
    for c in range(num_classes):
        p = (pred == c).float(); t = (target == c).float()
        scores.append((2 * (p * t).sum() / (p.sum() + t.sum() + eps)).item())
    return float(np.mean(scores))


# ── Training loops ────────────────────────────────────────────────────────────

def train_classification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_ds = OxfordIIITPetDataset(args.data, split="train")
    val_ds   = OxfordIIITPetDataset(args.data, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)

    # Use label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Lower LR with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Warmup for 5 epochs then cosine decay
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (args.epochs - warmup)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    wandb.init(project=args.wandb_project, name=f"cls_{args.run_name}",
               config=vars(args))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; correct = 0; total = 0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        scheduler.step()
        train_loss /= total
        train_acc   = correct / total

        model.eval()
        val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = nn.CrossEntropyLoss()(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        wandb.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                   "val/loss": val_loss, "val/acc": val_acc,
                   "lr": optimizer.param_groups[0]["lr"]})

        print(f"[Cls] Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_acc}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt, "checkpoints/classifier.pth")

    wandb.finish()
    print(f"Best val accuracy: {best_acc:.4f}")

def train_localization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_ds = OxfordIIITPetDataset(args.data, split="train")
    val_ds   = OxfordIIITPetDataset(args.data, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout).to(device)

    # Optionally load pretrained encoder from classifier
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        from models.classification import VGG11Classifier
        cls_model = VGG11Classifier(num_classes=37)
        ckpt = torch.load(args.pretrained_cls, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        cls_model.load_state_dict(sd)
        model.encoder.load_state_dict(cls_model.encoder.state_dict())
        print("Loaded pretrained encoder from classifier checkpoint.")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.wandb_project, name=f"loc_{args.run_name}",
               config=vars(args))

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0; total = 0
        for batch in train_loader:
            imgs  = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)

        scheduler.step()
        train_loss = total_loss / total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0; mean_iou = 0.0; val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                preds  = model(imgs)
                loss   = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
                val_loss  += loss.item() * imgs.size(0)
                mean_iou  += compute_iou_batch(preds, bboxes).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        mean_iou  /= val_total

        wandb.log({"epoch": epoch, "train/loss": train_loss,
                   "val/loss": val_loss, "val/iou": mean_iou,
                   "lr": optimizer.param_groups[0]["lr"]})

        print(f"[Loc] Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} IoU {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_iou}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt, "checkpoints/localizer.pth")

    wandb.finish()
    print(f"Best mean IoU: {best_iou:.4f}")


def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_ds = OxfordIIITPetDataset(args.data, split="train")
    val_ds   = OxfordIIITPetDataset(args.data, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout).to(device)

    # Optionally load pretrained encoder from classifier
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        from models.classification import VGG11Classifier
        cls_model = VGG11Classifier(num_classes=37)
        ckpt = torch.load(args.pretrained_cls, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        cls_model.load_state_dict(sd)
        model.encoder.load_state_dict(cls_model.encoder.state_dict())
        print("Loaded pretrained encoder from classifier checkpoint.")

        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen.")

    # Class weights to handle imbalance (bg >> fg >> uncertain)
    # Rough pixel distribution: bg~60%, fg~35%, uncertain~5%
    weights = torch.tensor([0.5, 1.5, 3.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.wandb_project, name=f"seg_{args.run_name}",
               config=vars(args))

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0; total = 0
        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)

        scheduler.step()
        train_loss = total_loss / total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0; val_dice = 0.0; val_pix_acc = 0.0; val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, masks)
                val_loss     += loss.item() * imgs.size(0)
                val_dice     += dice_score(logits, masks) * imgs.size(0)
                pix_acc       = (logits.argmax(1) == masks).float().mean().item()
                val_pix_acc  += pix_acc * imgs.size(0)
                val_total    += imgs.size(0)

        val_loss    /= val_total
        val_dice    /= val_total
        val_pix_acc /= val_total

        wandb.log({"epoch": epoch, "train/loss": train_loss,
                   "val/loss": val_loss, "val/dice": val_dice,
                   "val/pixel_acc": val_pix_acc,
                   "lr": optimizer.param_groups[0]["lr"]})

        print(f"[Seg] Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} Dice {val_dice:.4f} PixAcc {val_pix_acc:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_dice}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt, "checkpoints/unet.pth")

    wandb.finish()
    print(f"Best Dice: {best_dice:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 training")

    p.add_argument("--task", default="classification",
                   choices=["classification", "localization", "segmentation"])

    p.add_argument("--data", default="/autograder/data")

    p.add_argument("--epochs", type=int, default=60)

    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--workers", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb_project", default="da6401_a2")
    p.add_argument("--run_name", default="run")

    p.add_argument("--pretrained_cls", default="checkpoints/classifier.pth")

    p.add_argument("--freeze_encoder", action="store_true")

    return p.parse_args()

print("TASK:", args.task)
print("DATA:", args.data)
print("EPOCHS:", args.epochs)

if not os.path.exists(args.data):
    print("Switching to /autograder/data")
    args.data = "/autograder/data"


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    elif args.task == "localization":
        train_localization(args)
    elif args.task == "segmentation":
        train_segmentation(args)