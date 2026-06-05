"""
Predictor — Singleton model loader for the API.
Loads models once at startup, reuses for all requests.
"""

import sys
from pathlib import Path
from functools import lru_cache

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config
from src.utils.logger        import get_logger
from src.models.cnn          import build_cnn
from src.models.lstm         import build_lstm
from src.models.ensemble     import build_ensemble, BiometricsMLP
from src.features.mfcc              import extract_mfcc
from src.features.mel_spectrogram   import extract_mel_spectrogram
from src.features.voice_biometrics  import extract_voice_biometrics

import numpy as np

logger = get_logger("predictor")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class Predictor:
    """
    Loads all model weights once and exposes a single predict() method.
    Thread-safe for concurrent FastAPI requests (inference is read-only).
    """

    def __init__(self):
        self.cfg      = get_config()
        self.device   = torch.device("cpu")
        self.ensemble = None
        self._ready   = False
        self._load_models()

    def _load_model_weights(self, model, path: Path, name: str):
        if not path.exists():
            logger.warning(f"No weights at {path} — {name} uses random weights")
            return model
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded {name}")
        return model

    def _load_models(self):
        try:
            weights = self.cfg.paths.model_weights
            cnn     = self._load_model_weights(build_cnn(self.cfg),  weights / "cnn_best.pt",     "CNN")
            lstm    = self._load_model_weights(build_lstm(self.cfg), weights / "lstm_best.pt",    "LSTM")
            bio     = self._load_model_weights(
                BiometricsMLP(22, 2, 0.3), weights / "bio_mlp_best.pt", "BiometricsMLP"
            )
            self.ensemble          = build_ensemble(self.cfg)
            self.ensemble.cnn      = cnn
            self.ensemble.lstm     = lstm
            self.ensemble.bio_mlp  = bio
            self.ensemble.eval()
            self._ready = True
            logger.info("All models loaded and ready ✅")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    @torch.no_grad()
    def predict(self, audio_path: str, threshold: float = 0.65) -> dict:
        if not HAS_LIBROSA:
            raise RuntimeError("librosa not installed. Run: pip install librosa")

        cfg = self.cfg
        sr  = cfg.data.sample_rate

        # Load and normalize audio
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio / peak

        duration = len(audio) / sr

        # Extract features
        mel  = extract_mel_spectrogram(audio, sr, n_fft=cfg.data.n_fft, hop_length=cfg.data.hop_length)
        mfcc = extract_mfcc(audio, sr, n_mfcc=cfg.data.n_mfcc, n_fft=cfg.data.n_fft, hop_length=cfg.data.hop_length)
        bio  = extract_voice_biometrics(audio, sr, n_fft=cfg.data.n_fft, hop_length=cfg.data.hop_length)

        # To tensors — add batch dim
        t_mel  = torch.tensor(mel[np.newaxis],  dtype=torch.float32)
        t_mfcc = torch.tensor(mfcc[np.newaxis], dtype=torch.float32)
        t_bio  = torch.tensor(bio[np.newaxis],  dtype=torch.float32)

        self.ensemble.threshold = threshold
        result = self.ensemble(t_mel, t_mfcc, t_bio)

        return {
            "verdict":      "FAKE" if result["verdict"].item() == 1 else "REAL",
            "confidence":   round(result["proba_fake"].item(), 4),
            "proba_cnn":    round(result["proba_cnn"].item(),  4),
            "proba_lstm":   round(result["proba_lstm"].item(), 4),
            "proba_bio":    round(result["proba_bio"].item(),  4),
            "duration_sec": round(duration, 2),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Return singleton Predictor instance — loaded once, reused forever."""
    return Predictor()