"""
LSTM Model — MFCC Sequence Classifier
=======================================
Processes MFCC features as a time sequence and learns temporal patterns
that reveal AI generation artifacts.

What it detects temporally:
  - Unnatural smoothness in MFCC trajectories (TTS is too consistent)
  - Missing coarticulation (real speech slurs between phonemes)
  - Robotic rhythm — AI voices lack natural timing micro-variations
  - Prosody artifacts — unnatural pitch/energy transitions over time

Input shape:  (batch, 120, 128)  — 120 features × 128 time frames
The LSTM sees this as a sequence of 128 time steps,
each step being a 120-dim feature vector.

Architecture:
  Input (B, 120, 128)
    → Transpose → (B, 128, 120)   time-first for LSTM
    → LayerNorm on features
    → LSTM (3 layers, hidden=256, bidirectional)
    → Take last hidden state → (B, 512)   (256 × 2 directions)
    → Dropout + FC → (B, 2)   logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for MFCC sequence deepfake detection.

    Args:
        input_size:   Feature dimension per time step (3 * n_mfcc = 120)
        hidden_size:  LSTM hidden units per direction
        num_layers:   Number of stacked LSTM layers
        num_classes:  2 for binary classification
        dropout:      Dropout between LSTM layers and before FC
    """

    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Normalize input features (stabilises training with MFCC range)
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # expects (B, T, F)
            bidirectional=True,      # forward + backward = 2x hidden_size
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention — learn which time steps matter most
        self.attention = nn.Linear(hidden_size * 2, 1)

        lstm_out_size = hidden_size * 2   # bidirectional

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 120, 128)
               120 = feature dim, 128 = time frames
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Reshape: (B, features, time) → (B, time, features)
        x = x.permute(0, 2, 1)           # (B, 128, 120)

        # Normalise feature dimension
        x = self.input_norm(x)            # (B, 128, 120)

        # LSTM — output all hidden states
        out, _ = self.lstm(x)             # (B, 128, 512)

        # Attention pooling — weighted sum over time steps
        attn_weights = F.softmax(self.attention(out), dim=1)   # (B, 128, 1)
        context      = (out * attn_weights).sum(dim=1)          # (B, 512)

        return self.classifier(context)   # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def build_lstm(cfg) -> LSTMClassifier:
    return LSTMClassifier(
       input_size=120,
        hidden_size=cfg.model.lstm.hidden_size,
        num_layers=cfg.model.lstm.num_layers,
        num_classes=2,
        dropout=cfg.model.lstm.dropout,
    )