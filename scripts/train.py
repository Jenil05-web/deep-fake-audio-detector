"""
Step 6 — Train All Models
===========================
Trains CNN → LSTM → BiometricsMLP in sequence, then saves all weights.

Run:
    python scripts/train.py              # train all three
    python scripts/train.py --model cnn  # train only CNN
    python scripts/train.py --model lstm
    python scripts/train.py --model bio
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import BiometricsMLP
from src.training.dataset import get_dataloaders
from src.training.trainer import train_model

logger = get_logger("train")


def save_history(history: dict, name: str, cfg) -> None:
    logs_dir = Path(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{name}_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved → {path}")


def run(model_name: str = "all") -> None:
    cfg = get_config()
    set_seed(42)

    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)} | "
                f"Val batches: {len(val_loader)} | "
                f"Test batches: {len(test_loader)}")

    # ── CNN ───────────────────────────────────────────────────────────────────
    if model_name in ("all", "cnn"):
        logger.info("\n" + "="*50)
        logger.info("TRAINING CNN (Mel Spectrogram)")
        logger.info("="*50)
        cnn     = build_cnn(cfg)
        history = train_model(
            model=cnn,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            input_key="mel",
            model_name="cnn",
        )
        save_history(history, "cnn", cfg)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if model_name in ("all", "lstm"):
        logger.info("\n" + "="*50)
        logger.info("TRAINING LSTM (MFCC Sequences)")
        logger.info("="*50)
        lstm    = build_lstm(cfg)
        history = train_model(
            model=lstm,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            input_key="mfcc",
            model_name="lstm",
        )
        save_history(history, "lstm", cfg)

    # ── BiometricsMLP ─────────────────────────────────────────────────────────
    if model_name in ("all", "bio"):
        logger.info("\n" + "="*50)
        logger.info("TRAINING BiometricsMLP (Voice Features)")
        logger.info("="*50)
        bio_mlp = BiometricsMLP(input_dim=22, num_classes=2, dropout=0.3)
        history = train_model(
            model=bio_mlp,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            input_key="biometrics",
            model_name="bio_mlp",
        )
        save_history(history, "bio_mlp", cfg)

    logger.info("\n✅ Training complete!")
    logger.info(f"Weights saved in: {cfg.paths.model_weights}")
    logger.info("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "cnn", "lstm", "bio"],
                        default="all", help="Which model to train")
    args = parser.parse_args()
    run(args.model)