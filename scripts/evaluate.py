"""
Step 7 — Ensemble Evaluation
==============================
Loads the three trained model weights, runs the full ensemble
on the test set, and prints a complete evaluation report.

Run:
    python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import build_ensemble, BiometricsMLP
from src.training.dataset import get_dataloaders
from src.evaluation.metrics import compute_all_metrics

logger = get_logger("evaluate")


def load_weights(model, path: Path, model_name: str):
    """Load best checkpoint weights into model."""
    if not path.exists():
        logger.warning(f"No checkpoint found for {model_name} at {path}")
        logger.warning("Using random weights — train first with: python scripts/train.py")
        return model
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded {model_name} — epoch={ckpt.get('epoch','?')} "
                f"val_loss={ckpt.get('val_loss',0):.4f} "
                f"val_acc={ckpt.get('val_acc',0):.3f}")
    return model


@torch.no_grad()
def run_ensemble_on_loader(ensemble, loader, device) -> tuple:
    """Run ensemble over a DataLoader. Returns (all_probs, all_labels)."""
    ensemble.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        mel        = batch["mel"].to(device)
        mfcc       = batch["mfcc"].to(device)
        biometrics = batch["biometrics"].to(device)
        labels     = batch["label"]

        result = ensemble(mel, mfcc, biometrics)

        all_probs.extend(result["proba_fake"].cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_probs, all_labels


def print_report(metrics: dict, split: str) -> None:
    print(f"\n  {split.upper()} SET RESULTS")
    print(f"  {'─'*35}")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  AUC      : {metrics['auc']:.4f}  (1.0 = perfect)")
    print(f"  EER      : {metrics['eer']*100:.2f}%   (0% = perfect)")
    print(f"  F1 Score : {metrics['f1']:.4f}")


def run() -> None:
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Build and load models ─────────────────────────────────────────────────
    weights_dir = cfg.paths.model_weights

    cnn = load_weights(build_cnn(cfg),
                       weights_dir / "cnn_best.pt", "CNN")

    lstm = load_weights(build_lstm(cfg),
                        weights_dir / "lstm_best.pt", "LSTM")

    bio_mlp = load_weights(
        BiometricsMLP(input_dim=22, num_classes=2, dropout=0.3),
        weights_dir / "bio_mlp_best.pt", "BiometricsMLP"
    )

    # ── Build ensemble ────────────────────────────────────────────────────────
    ensemble = build_ensemble(cfg)
    ensemble.cnn     = cnn
    ensemble.lstm    = lstm
    ensemble.bio_mlp = bio_mlp
    ensemble = ensemble.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  ENSEMBLE EVALUATION REPORT")
    print("="*50)

    results = {}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        probs, labels = run_ensemble_on_loader(ensemble, loader, device)
        metrics = compute_all_metrics(probs, labels, threshold=0.65)
        print_report(metrics, split_name)
        results[split_name] = metrics

        # Per-model breakdown on test set
        if split_name == "test":
            print(f"\n  PER-MODEL BREAKDOWN (test set)")
            print(f"  {'─'*35}")
            for model, key, name in [
                (cnn,     "mel",        "CNN    "),
                (lstm,    "mfcc",       "LSTM   "),
                (bio_mlp, "biometrics", "Bio MLP"),
            ]:
                p_list, l_list = [], []
                model.eval().to(device)
                with torch.no_grad():
                    for batch in loader:
                        x = batch[key].to(device)
                        logits = model(x)
                        proba  = F.softmax(logits, dim=1)[:, 1]
                        p_list.extend(proba.cpu().tolist())
                        l_list.extend(batch["label"].tolist())
                m = compute_all_metrics(p_list, l_list)
                print(f"  {name}  acc={m['accuracy']:.3f}  "
                      f"auc={m['auc']:.4f}  eer={m['eer']*100:.1f}%  f1={m['f1']:.3f}")

    print("\n" + "="*50)

    # ── Save results ──────────────────────────────────────────────────────────
    logs_dir = Path(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_path}")
    logger.info("Next: python scripts/infer.py <audio_file>")


if __name__ == "__main__":
    run()