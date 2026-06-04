"""
MFCC Feature Extractor
========================
Extracts Mel-Frequency Cepstral Coefficients from audio.

MFCCs capture the spectral envelope of speech — they represent
HOW the vocal tract is shaped, which differs between real voices
and TTS/voice-conversion systems.

What we extract per file:
  - n_mfcc coefficients per frame  (default 40)
  - Delta MFCCs   (velocity — how fast coefficients change)
  - Delta-Delta MFCCs (acceleration)
  Final shape: (3 * n_mfcc, T) where T = number of time frames
"""

import numpy as np
import librosa
from pathlib import Path


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fixed_length: int = 128,   # pad/clip to this many time frames
) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta from a raw audio array.

    Args:
        audio:        Raw waveform, float32, shape (N,)
        sr:           Sample rate (should be 16000)
        n_mfcc:       Number of MFCC coefficients
        n_fft:        FFT window size
        hop_length:   Hop size between frames
        n_mels:       Number of mel filter banks
        fixed_length: Pad or clip time axis to this length

    Returns:
        np.ndarray of shape (3 * n_mfcc, fixed_length), dtype float32
    """
    # Base MFCCs — shape: (n_mfcc, T)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Delta (1st order derivative) — captures temporal dynamics
    delta = librosa.feature.delta(mfcc, order=1)

    # Delta-delta (2nd order derivative) — captures acceleration
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack → shape: (3 * n_mfcc, T)
    features = np.vstack([mfcc, delta, delta2])

    # Pad or clip to fixed_length along time axis
    features = _pad_or_clip(features, fixed_length)

    # Per-feature mean/std normalisation (across time axis)
    features = _normalize(features)

    return features.astype(np.float32)


def _pad_or_clip(features: np.ndarray, fixed_length: int) -> np.ndarray:
    """Pad with zeros or clip along time axis (axis=1)."""
    T = features.shape[1]
    if T < fixed_length:
        pad = fixed_length - T
        features = np.pad(features, ((0, 0), (0, pad)), mode="constant")
    else:
        features = features[:, :fixed_length]
    return features


def _normalize(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per coefficient (row-wise)."""
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    return (features - mean) / std


def extract_mfcc_from_file(
    path: Path,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    fixed_length: int = 128,
) -> np.ndarray:
    """Convenience wrapper: load file then extract MFCCs."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length, fixed_length=fixed_length)