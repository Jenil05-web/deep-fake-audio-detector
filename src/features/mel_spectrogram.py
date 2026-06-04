"""
Mel Spectrogram Feature Extractor
====================================
Converts raw audio into a 2D image-like representation.

The CNN model treats this as a single-channel image and learns
visual patterns — smearing, over-smoothness, unnatural harmonics —
that are characteristic of AI-generated voices.

Output shape: (1, n_mels, fixed_length) — ready to feed into a CNN.
"""

import numpy as np
import librosa
from pathlib import Path


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fixed_length: int = 128,   # time frames
    top_db: float = 80.0,      # dynamic range for log compression
) -> np.ndarray:
    """
    Extract log-Mel spectrogram from raw audio.

    Args:
        audio:        Raw waveform, float32, shape (N,)
        sr:           Sample rate
        n_fft:        FFT window size
        hop_length:   Hop size between frames
        n_mels:       Number of mel filter banks (frequency bins)
        fixed_length: Pad or clip time axis to this many frames
        top_db:       Dynamic range cutoff for log compression

    Returns:
        np.ndarray of shape (1, n_mels, fixed_length), dtype float32
        The leading 1 is the channel dimension expected by PyTorch CNNs.
    """
    # Compute mel spectrogram — shape: (n_mels, T)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Convert to log scale (dB) — more perceptually meaningful
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)

    # Pad or clip time axis
    mel_db = _pad_or_clip(mel_db, fixed_length)

    # Normalize to [0, 1]
    mel_db = _minmax_normalize(mel_db)

    # Add channel dim → (1, n_mels, fixed_length)
    mel_db = mel_db[np.newaxis, :, :]

    return mel_db.astype(np.float32)


def _pad_or_clip(spec: np.ndarray, fixed_length: int) -> np.ndarray:
    T = spec.shape[1]
    if T < fixed_length:
        pad = fixed_length - T
        spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant", constant_values=spec.min())
    else:
        spec = spec[:, :fixed_length]
    return spec


def _minmax_normalize(spec: np.ndarray) -> np.ndarray:
    min_val = spec.min()
    max_val = spec.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(spec)
    return (spec - min_val) / (max_val - min_val)


def extract_mel_from_file(
    path: Path,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fixed_length: int = 128,
) -> np.ndarray:
    """Convenience wrapper: load file then extract Mel spectrogram."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return extract_mel_spectrogram(audio, sr, n_fft, hop_length, n_mels, fixed_length)