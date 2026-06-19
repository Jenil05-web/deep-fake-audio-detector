"""
CNN Model — Mel Spectrogram Classifier
========================================
Treats the mel spectrogram as a single-channel image (1 x 128 x 128)
and learns visual patterns that distinguish real vs AI-generated audio.

What it detects visually:
  - Over-smoothness in TTS spectrograms (vocoders blur fine structure)
  - Unnatural harmonic regularity (AI voices are "too perfect")
  - Vocoder artifacts — grid-like patterns from synthesis frames
  - Missing micro-variations in frequency that real speech has

Architecture:
  Input (1, 128, 128)
    → Conv Block 1: 32 filters  → (32, 64, 64)
    → Conv Block 2: 64 filters  → (64, 32, 32)
    → Conv Block 3: 128 filters → (128, 16, 16)
    → Conv Block 4: 256 filters → (256, 8, 8)
    → Global Average Pooling    → (256,)
    → Dropout + FC              → (2,)   logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32,  pool=True),
            ConvBlock(32,          64,  pool=True),
            ConvBlock(64,          128, pool=True),
            ConvBlock(128,         256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def build_cnn(cfg) -> CNNClassifier:
    return CNNClassifier(
        in_channels=1,
        num_classes=2,
        dropout=cfg.model.cnn.dropout,
    )