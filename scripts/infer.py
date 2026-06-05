"""
Step 7b — Single File Inference
=================================
Run the full ensemble on one audio file and get a verdict.

This is the core function the API will call in Step 9.

Run:
    python scripts/infer.py path/to/audio.wav
    python scripts/infer.py path/to/audio.wav --threshold 0.7
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.cnn      import build_cnn
from src.models.lstm     import build_lstm
from src.models.ensemble import build_ensemble, BiometricsMLP
from src.features.mfcc             import extract_mfcc
from src.features.mel_spectrogram  import extract_mel_spectrogram
from src.features.voice_biometrics import extract_voice_biometrics

logger = get_logger("infer")

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Run: pip install librosa soundfile")
    sys.exit(1)


def load_weights(model, path: Path, name: str):
    if not path.exists():
        logger.warning(f"No weights at {path} — using random weights for {name}")
        return model
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return model


def preprocess_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Load, resample, mono, peak-normalize."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak
    return audio


def extract_features(audio: np.ndarray, cfg) -> dict:
    """Run all three feature extractors and return batch-ready tensors."""
    sr         = cfg.data.sample_rate
    n_fft      = cfg.data.n_fft
    hop_length = cfg.data.hop_length
    n_mfcc     = cfg.data.n_mfcc

    mel  = extract_mel_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)
    mfcc = extract_mfcc(audio, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    bio  = extract_voice_biometrics(audio, sr, n_fft=n_fft, hop_length=hop_length)

    return {
        "mel":        torch.tensor(mel[np.newaxis],  dtype=torch.float32),  # (1,1,128,128)
        "mfcc":       torch.tensor(mfcc[np.newaxis], dtype=torch.float32),  # (1,120,128)
        "biometrics": torch.tensor(bio[np.newaxis],  dtype=torch.float32),  # (1,22)
    }


def build_inference_ensemble(cfg, threshold: float):
    """Build ensemble and load all three trained weights."""
    weights_dir = cfg.paths.model_weights

    cnn     = load_weights(build_cnn(cfg),  weights_dir / "cnn_best.pt",     "CNN")
    lstm    = load_weights(build_lstm(cfg), weights_dir / "lstm_best.pt",    "LSTM")
    bio_mlp = load_weights(
        BiometricsMLP(input_dim=22, num_classes=2, dropout=0.3),
        weights_dir / "bio_mlp_best.pt", "BiometricsMLP"
    )

    ensemble          = build_ensemble(cfg)
    ensemble.cnn      = cnn
    ensemble.lstm     = lstm
    ensemble.bio_mlp  = bio_mlp
    ensemble.threshold = threshold
    ensemble.eval()
    return ensemble


@torch.no_grad()
def predict(audio_path: str, threshold: float = 0.65) -> dict:
    """
    Full inference pipeline for one audio file.

    Returns:
        {
            "verdict":      "FAKE" or "REAL",
            "confidence":   0.0 - 1.0  (probability of fake),
            "proba_cnn":    float,
            "proba_lstm":   float,
            "proba_bio":    float,
            "duration_sec": float,
            "inference_ms": float,
        }
    """
    cfg      = get_config()
    path     = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    t_start  = time.time()

    # Load model
    ensemble = build_inference_ensemble(cfg, threshold)

    # Preprocess audio
    audio    = preprocess_audio(path, cfg.data.sample_rate)
    duration = len(audio) / cfg.data.sample_rate

    # Extract features
    features = extract_features(audio, cfg)

    # Run ensemble
    result   = ensemble(
        features["mel"],
        features["mfcc"],
        features["biometrics"],
    )

    inference_ms = (time.time() - t_start) * 1000

    verdict    = "FAKE" if result["verdict"].item() == 1 else "REAL"
    confidence = result["proba_fake"].item()

    return {
        "verdict":      verdict,
        "confidence":   round(confidence, 4),
        "proba_cnn":    round(result["proba_cnn"].item(),  4),
        "proba_lstm":   round(result["proba_lstm"].item(), 4),
        "proba_bio":    round(result["proba_bio"].item(),  4),
        "duration_sec": round(duration, 2),
        "inference_ms": round(inference_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Deepfake audio inference")
    parser.add_argument("audio", help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Fake probability threshold (default: 0.65)")
    args = parser.parse_args()

    result = predict(args.audio, args.threshold)

    # Pretty print
    verdict_label = f"🔴 {result['verdict']}" if result['verdict'] == "FAKE" else f"🟢 {result['verdict']}"
    print("\n" + "="*45)
    print("  DEEPFAKE AUDIO DETECTION RESULT")
    print("="*45)
    print(f"  File       : {args.audio}")
    print(f"  Duration   : {result['duration_sec']}s")
    print(f"  Verdict    : {verdict_label}")
    print(f"  Confidence : {result['confidence']*100:.1f}% fake")
    print(f"  ─────────────────────────────────────")
    print(f"  CNN score  : {result['proba_cnn']*100:.1f}%")
    print(f"  LSTM score : {result['proba_lstm']*100:.1f}%")
    print(f"  Bio score  : {result['proba_bio']*100:.1f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Inference  : {result['inference_ms']}ms")
    print("="*45 + "\n")


if __name__ == "__main__":
    main()