"""
Step 8 — Export Models to ONNX
================================
Exports CNN, LSTM, and BiometricsMLP to ONNX format.

Why ONNX?
  - 3-5x faster inference than PyTorch (no Python overhead)
  - No PyTorch dependency needed at serving time
  - Works with ONNX Runtime on CPU/GPU
  - Portable — can be deployed anywhere

Output:
    models/weights/cnn.onnx
    models/weights/lstm.onnx
    models/weights/bio_mlp.onnx

Run:
    python scripts/export_onnx.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import BiometricsMLP

logger = get_logger("export_onnx")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("onnx/onnxruntime not installed.")
    logger.warning("Run: pip install onnx onnxruntime")


def load_weights(model: nn.Module, path: Path, name: str) -> nn.Module:
    if not path.exists():
        logger.warning(f"No checkpoint at {path} — exporting with random weights for {name}")
        return model
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded {name} weights from {path.name}")
    return model


def export_model(
    model:      nn.Module,
    dummy_input,
    out_path:   Path,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    model_name:  str,
) -> bool:
    """Export one model to ONNX and verify it."""
    model.eval()

    logger.info(f"Exporting {model_name} → {out_path.name}")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,   # fold constants for speed
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # variable batch size
    )

    if not HAS_ONNX:
        logger.info(f"  Exported (verification skipped — install onnx to verify)")
        return True

    # ── Verify ONNX model structure ───────────────────────────────────────────
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"  ✅ ONNX structure valid")

    # ── Verify ONNX Runtime output matches PyTorch output ────────────────────
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    # Run PyTorch
    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            pt_out = model(*dummy_input).numpy()
            ort_inputs = {
                name: inp.numpy()
                for name, inp in zip(input_names, dummy_input)
            }
        else:
            pt_out = model(dummy_input).numpy()
            ort_inputs = {input_names[0]: dummy_input.numpy()}

    # Run ONNX Runtime
    ort_out = sess.run(None, ort_inputs)[0]

    import numpy as np
    max_diff = abs(pt_out - ort_out).max()
    logger.info(f"  Max output diff PyTorch vs ONNX: {max_diff:.2e}  "
                f"({'✅ OK' if max_diff < 1e-4 else '⚠️ High'})")

    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"  File size: {size_mb:.1f} MB")
    return True


def run() -> None:
    cfg         = get_config()
    weights_dir = cfg.paths.model_weights
    weights_dir.mkdir(parents=True, exist_ok=True)

    B = 1   # batch size for export (dynamic axes allow any size at runtime)

    print("\n" + "="*50)
    print("  ONNX EXPORT")
    print("="*50)

    # ── CNN ───────────────────────────────────────────────────────────────────
    cnn = load_weights(build_cnn(cfg), weights_dir / "cnn_best.pt", "CNN")
    export_model(
        model        = cnn,
        dummy_input  = torch.randn(B, 1, 128, 128),
        out_path     = weights_dir / "cnn.onnx",
        input_names  = ["mel"],
        output_names = ["logits"],
        dynamic_axes = {"mel": {0: "batch"}, "logits": {0: "batch"}},
        model_name   = "CNN",
    )

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm = load_weights(build_lstm(cfg), weights_dir / "lstm_best.pt", "LSTM")
    export_model(
        model        = lstm,
        dummy_input  = torch.randn(B, 120, 128),
        out_path     = weights_dir / "lstm.onnx",
        input_names  = ["mfcc"],
        output_names = ["logits"],
        dynamic_axes = {"mfcc": {0: "batch"}, "logits": {0: "batch"}},
        model_name   = "LSTM",
    )

    # ── BiometricsMLP ─────────────────────────────────────────────────────────
    bio_mlp = load_weights(
        BiometricsMLP(input_dim=22, num_classes=2, dropout=0.3),
        weights_dir / "bio_mlp_best.pt", "BiometricsMLP"
    )
    export_model(
        model        = bio_mlp,
        dummy_input  = torch.randn(B, 22),
        out_path     = weights_dir / "bio_mlp.onnx",
        input_names  = ["biometrics"],
        output_names = ["logits"],
        dynamic_axes = {"biometrics": {0: "batch"}, "logits": {0: "batch"}},
        model_name   = "BiometricsMLP",
    )

    print("\n" + "="*50)
    print("  ✅ Export complete!")
    print(f"  Files in: {weights_dir}")
    print("  Next: python scripts/benchmark_onnx.py")
    print("="*50 + "\n")


if __name__ == "__main__":
    run()