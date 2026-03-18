"""
Training loop for ECG Risk Stratification
Supports: 1D ResNet-34 (extendable to CNN-Transformer)
Logs: train/val loss, macro F1, per-class F1 each epoch
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders, get_class_weights
from models.resnet1d import ResNet1D

# ── Config ─────────────────────────────────────────────────
CONFIG = {
    "data_dir"    : "data/ptbxl/records/",
    "split_dir"   : "data/",
    "save_dir"    : "checkpoints/",
    "batch_size"  : 32,
    "epochs"      : 30,
    "lr"          : 3e-4,
    "weight_decay": 1e-4,
    "dropout"     : 0.3,
    "patience"    : 7,
    "num_classes" : 3,
    "seed"        : 42,
}

CLASSES = ["Low", "Medium", "High"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── Data ───────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        CONFIG["data_dir"], CONFIG["split_dir"], CONFIG["batch_size"]
    )

    # ── Model ──────────────────────────────────────────────
    model = ResNet1D(num_classes=CONFIG["num_classes"],
                     dropout=CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet1D | Parameters: {total_params:,}\n")

    # ── Loss with class weights ────────────────────────────
    class_weights = get_class_weights(
        os.path.join(CONFIG["split_dir"], "train.csv")
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & Scheduler ──────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # ── Training loop ──────────────────────────────────────
    best_val_f1  = 0.0
    patience_ctr = 0
    history      = []

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train F1':>9} {'Val Loss':>9} {'Val F1':>7} {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )
        scheduler.step()

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_f1": train_f1, "val_loss": val_loss, "val_f1": val_f1
        })

        print(f"{epoch:>5} {train_loss:>11.4f} {train_f1:>9.4f} "
              f"{val_loss:>9.4f} {val_f1:>7.4f} {elapsed:>5.1f}s")

        # ── Save best model ────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(CONFIG["save_dir"], "resnet1d_best.pt"))
            print(f"  ✓ Saved best model (val F1: {best_val_f1:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Final evaluation on test set ───────────────────────
    print("\n── Test Set Evaluation ──────────────────────────")
    model.load_state_dict(
        torch.load(os.path.join(CONFIG["save_dir"], "resnet1d_best.pt"),
                   map_location=device)
    )
    _, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(f"Test Macro F1: {test_f1:.4f}")
    print("\nPer-class Report:")
    print(classification_report(test_labels, test_preds,
                                 target_names=CLASSES, digits=4))

    os.makedirs("results", exist_ok=True)
    with open("results/resnet1d_results.txt", "w") as f:
        f.write(f"Best Val F1 : {best_val_f1:.4f}\n")
        f.write(f"Test Macro F1: {test_f1:.4f}\n\n")
        f.write(classification_report(test_labels, test_preds,
                                       target_names=CLASSES, digits=4))
    print("\nResults saved to results/resnet1d_results.txt")


if __name__ == "__main__":
    main()