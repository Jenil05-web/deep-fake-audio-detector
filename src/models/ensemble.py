"""
Ensemble Model
================
Combines CNN + LSTM + Biometrics MLP into one unified model.

Why ensemble?
  - CNN catches spectral/visual artifacts in mel spectrograms
  - LSTM catches temporal/rhythmic artifacts in MFCC sequences
  - Biometrics MLP catches voice quality anomalies (jitter, shimmer, HNR)
  - Together they cover different attack types that fool individual models

Architecture:
  mel        → CNN    → (B, 2) logits_cnn
  mfcc       → LSTM   → (B, 2) logits_lstm
  biometrics → MLP    → (B, 2) logits_bio

  Weighted average of probabilities → final confidence score
  Threshold (default 0.65) → verdict: real / fake

Two modes:
  1. Independent training — train CNN and LSTM separately first
  2. Joint inference     — load both weights, fuse at prediction time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn  import CNNClassifier
from src.models.lstm import LSTMClassifier


class BiometricsMLP(nn.Module):
    """
    Small MLP for the 22-dim voice biometrics vector.
    Fast to train — biometrics are already engineered features.
    """

    def __init__(self, input_dim: int = 22, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleModel(nn.Module):
    """
    Full ensemble: CNN + LSTM + BiometricsMLP.

    Args:
        cnn:      Trained CNNClassifier
        lstm:     Trained LSTMClassifier
        bio_mlp:  BiometricsMLP
        weights:  [w_cnn, w_lstm, w_bio] — must sum to 1.0
        threshold: Probability threshold for fake verdict (default 0.65)
    """

    def __init__(
        self,
        cnn:       CNNClassifier,
        lstm:      LSTMClassifier,
        bio_mlp:   BiometricsMLP,
        weights:   list = None,
        threshold: float = 0.65,
    ):
        super().__init__()
        self.cnn       = cnn
        self.lstm      = lstm
        self.bio_mlp   = bio_mlp
        self.threshold = threshold

        # Learnable fusion weights (initialised from config, then softmax-normalised)
        w = weights or [0.35, 0.35, 0.30]
        self.register_buffer(
            "weights",
            F.softmax(torch.tensor(w, dtype=torch.float32), dim=0),
        )

    def forward(
        self,
        mel:        torch.Tensor,   # (B, 1, 128, 128)
        mfcc:       torch.Tensor,   # (B, 120, 128)
        biometrics: torch.Tensor,   # (B, 22)
    ) -> dict:
        """
        Returns a dict with:
          proba_fake  : (B,)  probability of being fake  [0, 1]
          proba_real  : (B,)  probability of being real  [0, 1]
          verdict     : (B,)  1=fake, 0=real  (after threshold)
          proba_cnn   : (B,)  CNN contribution
          proba_lstm  : (B,)  LSTM contribution
          proba_bio   : (B,)  Biometrics contribution
        """
        # Get fake-class probability from each model
        p_cnn  = F.softmax(self.cnn(mel),             dim=1)[:, 1]   # (B,)
        p_lstm = F.softmax(self.lstm(mfcc),           dim=1)[:, 1]   # (B,)
        p_bio  = F.softmax(self.bio_mlp(biometrics),  dim=1)[:, 1]   # (B,)

        # Weighted fusion
        w = self.weights
        p_fake = w[0] * p_cnn + w[1] * p_lstm + w[2] * p_bio        # (B,)

        return {
            "proba_fake": p_fake,
            "proba_real": 1.0 - p_fake,
            "verdict":    (p_fake >= self.threshold).long(),
            "proba_cnn":  p_cnn,
            "proba_lstm": p_lstm,
            "proba_bio":  p_bio,
        }

    def predict(
        self,
        mel:        torch.Tensor,
        mfcc:       torch.Tensor,
        biometrics: torch.Tensor,
    ) -> tuple:
        """Convenience method. Returns (verdict, confidence_score)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(mel, mfcc, biometrics)
        return out["verdict"], out["proba_fake"]


def build_ensemble(cfg) -> EnsembleModel:
    """Build full ensemble from config."""
    from src.models.cnn  import build_cnn
    from src.models.lstm import build_lstm

    cnn     = build_cnn(cfg)
    lstm    = build_lstm(cfg)
    bio_mlp = BiometricsMLP(
        input_dim=22,
        num_classes=2,
        dropout=cfg.model.cnn.dropout,
    )

    return EnsembleModel(
        cnn=cnn,
        lstm=lstm,
        bio_mlp=bio_mlp,
        weights=[0.35, 0.35, 0.30],
        threshold=0.65,
    )