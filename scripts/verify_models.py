"""
Step 5b — Verify Models
========================
Confirms all three models instantiate correctly and produce
the right output shapes before we start training.

Run:
    python scripts/verify_models.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import build_ensemble, BiometricsMLP

logger = get_logger("verify_models")


def count_params(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{train:,} trainable / {total:,} total"


def verify() -> None:
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B      = 4   # dummy batch size

    print("\n" + "=" * 58)
    print("  MODEL VERIFICATION")
    print(f"  Device: {device}")
    print("=" * 58)

    # Dummy inputs matching real data shapes
    mel        = torch.randn(B, 1, 128, 128).to(device)
    mfcc       = torch.randn(B, 120, 128).to(device)
    biometrics = torch.randn(B, 22).to(device)

    # ── CNN ───────────────────────────────────────────────────────────────────
    print("\n  [1] CNN Classifier")
    cnn = build_cnn(cfg).to(device)
    cnn.eval()
    with torch.no_grad():
        out = cnn(mel)
    print(f"    Input  : {tuple(mel.shape)}")
    print(f"    Output : {tuple(out.shape)}  (expected ({B}, 2))")
    print(f"    Params : {count_params(cnn)}")
    assert out.shape == (B, 2), f"CNN output shape wrong: {out.shape}"
    print("    ✅ OK")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n  [2] LSTM Classifier")
    lstm = build_lstm(cfg).to(device)
    lstm.eval()
    with torch.no_grad():
        out = lstm(mfcc)
    print(f"    Input  : {tuple(mfcc.shape)}")
    print(f"    Output : {tuple(out.shape)}  (expected ({B}, 2))")
    print(f"    Params : {count_params(lstm)}")
    assert out.shape == (B, 2), f"LSTM output shape wrong: {out.shape}"
    print("    ✅ OK")

    # ── BiometricsMLP ─────────────────────────────────────────────────────────
    print("\n  [3] Biometrics MLP")
    bio = BiometricsMLP(input_dim=22, num_classes=2).to(device)
    bio.eval()
    with torch.no_grad():
        out = bio(biometrics)
    print(f"    Input  : {tuple(biometrics.shape)}")
    print(f"    Output : {tuple(out.shape)}  (expected ({B}, 2))")
    print(f"    Params : {count_params(bio)}")
    assert out.shape == (B, 2), f"MLP output shape wrong: {out.shape}"
    print("    ✅ OK")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n  [4] Ensemble (CNN + LSTM + MLP)")
    ensemble = build_ensemble(cfg).to(device)
    ensemble.eval()
    with torch.no_grad():
        result = ensemble(mel, mfcc, biometrics)
    print(f"    proba_fake : {result['proba_fake'].shape}  values: {result['proba_fake'].round(decimals=3).tolist()}")
    print(f"    proba_real : {result['proba_real'].shape}")
    print(f"    verdict    : {result['verdict'].tolist()}  (0=real, 1=fake)")
    print(f"    Params     : {count_params(ensemble)}")
    assert result["proba_fake"].shape == (B,)
    assert result["verdict"].shape    == (B,)
    print("    ✅ OK")

    print("\n" + "-" * 58)
    print("  ✅ All models verified! Ready for Step 6: Training Loop")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    verify()