"""
Step 6 — Training Loop
========================
Trains CNN and LSTM independently, then freezes them for ensemble use.

Training strategy:
  1. Train CNN  on mel spectrograms   → save best weights
  2. Train LSTM on MFCC sequences     → save best weights
  3. Ensemble uses both frozen models + trains BiometricsMLP on top

Features:
  - Early stopping (stops if val loss doesn't improve for N epochs)
  - Best model checkpointing (saves only the best val-loss checkpoint)
  - Per-epoch metrics: loss, accuracy, AUC
  - Runs on CPU or GPU automatically
"""

import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_auc(probs: list, labels: list) -> float:
    """Simple AUC via trapezoidal rule — no sklearn needed."""
    from src.evaluation.metrics import compute_auc as _auc
    return _auc(probs, labels)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Core train / eval loops ───────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    input_key:  str,          # "mel" or "mfcc"
) -> dict:
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch in loader:
        x      = batch[input_key].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, labels)
        n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc":  total_acc  / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    input_key: str,
) -> dict:
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        x      = batch[input_key].to(device)
        labels = batch["label"].to(device)

        logits = model(x)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # fake probability

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, labels)
        n_batches  += 1

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # AUC — gracefully skip if only one class in val set
    try:
        auc = compute_auc(all_probs, all_labels)
    except Exception:
        auc = 0.0

    return {
        "loss":   total_loss / max(n_batches, 1),
        "acc":    total_acc  / max(n_batches, 1),
        "auc":    auc,
        "probs":  all_probs,
        "labels": all_labels,
    }


# ── Main trainer ──────────────────────────────────────────────────────────────

def train_model(
    model:       nn.Module,
    train_loader: DataLoader,
    val_loader:  DataLoader,
    cfg,
    input_key:   str,          # "mel" for CNN, "mfcc" for LSTM, "biometrics" for MLP
    model_name:  str = "model",
    logger=None,
) -> dict:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        model:        CNN, LSTM, or BiometricsMLP instance
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        cfg:          AppConfig
        input_key:    Which batch key to feed into model
        model_name:   Used for checkpoint filename
        logger:       Optional logger instance

    Returns:
        history dict with train/val loss, acc, auc per epoch
    """
    from src.utils.logger import get_logger
    log = logger or get_logger(model_name)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=1e-4,
    )

    # Cosine annealing — gradually reduces LR to near zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    criterion    = nn.CrossEntropyLoss()
    early_stop   = EarlyStopping(patience=cfg.training.early_stopping_patience)

    weights_dir  = Path(cfg.paths.model_weights)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path    = weights_dir / f"{model_name}_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [],
               "val_acc": [], "val_auc": []}

    best_val_loss = float("inf")
    log.info(f"Training {model_name} on {device} for up to {cfg.training.epochs} epochs")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, input_key)
        val_metrics   = evaluate(model, val_loader, criterion, device, input_key)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:>3}/{cfg.training.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
            f"auc={val_metrics['auc']:.3f} | {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val_loss,
                "val_acc":     val_metrics["acc"],
                "val_auc":     val_metrics["auc"],
            }, best_path)
            log.info(f"  ✅ Saved best checkpoint → {best_path.name}")

        # Early stopping
        if early_stop.step(val_metrics["loss"]):
            log.info(f"  Early stopping at epoch {epoch}")
            break

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Best weights saved → {best_path}")
    history["best_val_loss"] = best_val_loss
    history["best_path"]     = str(best_path)
    return history