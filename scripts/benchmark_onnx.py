"""
Step 8b — Benchmark ONNX vs PyTorch
======================================
Compares inference speed between PyTorch and ONNX Runtime.
Helps confirm the ONNX export is worth using in the API.

Run:
    python scripts/benchmark_onnx.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import BiometricsMLP

logger = get_logger("benchmark")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def benchmark_pytorch(model, dummy_input, n_runs: int = 50) -> float:
    """Returns mean inference time in ms."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    # Timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    return (time.perf_counter() - t0) / n_runs * 1000


def benchmark_onnx(onnx_path: Path, input_name: str,
                   dummy_np: np.ndarray, n_runs: int = 50) -> float:
    """Returns mean inference time in ms."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: dummy_np})
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: dummy_np})
    return (time.perf_counter() - t0) / n_runs * 1000


def run() -> None:
    cfg         = get_config()
    weights_dir = cfg.paths.model_weights
    N_RUNS      = 50

    print("\n" + "="*55)
    print("  ONNX vs PyTorch Benchmark  (batch_size=1, CPU)")
    print("="*55)
    print(f"  {'Model':<16} {'PyTorch':>12} {'ONNX':>12} {'Speedup':>10}")
    print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*10}")

    benchmarks = [
        {
            "name":       "CNN",
            "model":      build_cnn(cfg),
            "weights":    weights_dir / "cnn_best.pt",
            "onnx":       weights_dir / "cnn.onnx",
            "input_name": "mel",
            "dummy_pt":   torch.randn(1, 1, 128, 128),
            "dummy_np":   np.random.randn(1, 1, 128, 128).astype(np.float32),
        },
        {
            "name":       "LSTM",
            "model":      build_lstm(cfg),
            "weights":    weights_dir / "lstm_best.pt",
            "onnx":       weights_dir / "lstm.onnx",
            "input_name": "mfcc",
            "dummy_pt":   torch.randn(1, 120, 128),
            "dummy_np":   np.random.randn(1, 120, 128).astype(np.float32),
        },
        {
            "name":       "BiometricsMLP",
            "model":      BiometricsMLP(22, 2, 0.3),
            "weights":    weights_dir / "bio_mlp_best.pt",
            "onnx":       weights_dir / "bio_mlp.onnx",
            "input_name": "biometrics",
            "dummy_pt":   torch.randn(1, 22),
            "dummy_np":   np.random.randn(1, 22).astype(np.float32),
        },
    ]

    total_pt, total_onnx = 0.0, 0.0

    for b in benchmarks:
        # Load PyTorch weights
        if b["weights"].exists():
            ckpt = torch.load(str(b["weights"]), map_location="cpu", weights_only=True)
            b["model"].load_state_dict(ckpt["model_state"])

        pt_ms = benchmark_pytorch(b["model"], b["dummy_pt"], N_RUNS)
        total_pt += pt_ms

        if HAS_ORT and b["onnx"].exists():
            onnx_ms = benchmark_onnx(b["onnx"], b["input_name"], b["dummy_np"], N_RUNS)
            total_onnx += onnx_ms
            speedup = pt_ms / max(onnx_ms, 0.001)
            print(f"  {b['name']:<16} {pt_ms:>10.2f}ms {onnx_ms:>10.2f}ms {speedup:>9.1f}x")
        else:
            msg = "not exported" if not b["onnx"].exists() else "ort missing"
            print(f"  {b['name']:<16} {pt_ms:>10.2f}ms {'─':>12} {msg:>10}")

    if HAS_ORT and total_onnx > 0:
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<16} {total_pt:>10.2f}ms {total_onnx:>10.2f}ms "
              f"{total_pt/total_onnx:>9.1f}x")

    print("="*55)

    if not HAS_ORT:
        print("\n  Install onnxruntime to see ONNX benchmarks:")
        print("  pip install onnxruntime\n")


if __name__ == "__main__":
    run()