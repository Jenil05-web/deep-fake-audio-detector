"""
Voice Biometrics Feature Extractor
=====================================
Extracts low-level voice quality metrics that expose AI-generation artifacts.

Why these features?
  - Real voices have natural micro-variations (jitter, shimmer)
  - TTS systems produce unnaturally smooth/periodic signals
  - GAN vocoders leave phase and spectral artifacts

Features extracted (all scalar → concatenated into 1D vector):
  ┌─────────────────────────────────────────────────────────────┐
  │ Pitch (F0)     mean, std, min, max, range                   │
  │ Jitter         freq variation cycle-to-cycle (aperiodicity) │
  │ Shimmer        amplitude variation cycle-to-cycle           │
  │ HNR            Harmonic-to-Noise Ratio                      │
  │ ZCR            Zero Crossing Rate stats                     │
  │ Spectral       centroid, bandwidth, rolloff, flatness       │
  │ RMS Energy     mean, std                                    │
  └─────────────────────────────────────────────────────────────┘
  Total: 22-dimensional vector per audio clip.
"""

import numpy as np
import librosa
from pathlib import Path


# ── Individual extractors ──────────────────────────────────────────────────────

def extract_pitch_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract F0 (fundamental frequency) statistics.
    Uses librosa's YIN algorithm — works well for speech.
    Returns 5 values: mean, std, min, max, range
    """
    # pyin gives more accurate pitch for speech than yin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz — below human voice floor
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — above human voice ceiling
            sr=sr,
        )
        # Only use voiced frames (where pitch was detected)
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) < 2:
        return np.zeros(5, dtype=np.float32)

    return np.array([
        f0_voiced.mean(),
        f0_voiced.std(),
        f0_voiced.min(),
        f0_voiced.max(),
        f0_voiced.max() - f0_voiced.min(),
    ], dtype=np.float32)


def extract_jitter(audio: np.ndarray, sr: int) -> float:
    """
    Jitter = cycle-to-cycle variation in pitch period.
    High jitter = natural voice. Very low jitter = TTS artifact.
    Approximated via F0 period variance.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception:
        return 0.0

    if len(f0_voiced) < 2:
        return 0.0

    periods = 1.0 / (f0_voiced + 1e-8)
    diffs   = np.abs(np.diff(periods))
    jitter  = diffs.mean() / (periods.mean() + 1e-8)
    return float(jitter)


def extract_shimmer(audio: np.ndarray, sr: int, hop_length: int = 512) -> float:
    """
    Shimmer = cycle-to-cycle variation in amplitude.
    Computed as mean absolute difference between consecutive RMS frames,
    normalised by mean RMS.
    """
    rms    = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    diffs  = np.abs(np.diff(rms))
    mean   = rms.mean() + 1e-8
    return float(diffs.mean() / mean)


def extract_hnr(audio: np.ndarray, sr: int) -> float:
    """
    Harmonic-to-Noise Ratio — approximated via autocorrelation.
    High HNR = clear/periodic voice. Low HNR = noisy or synthetic.
    """
    # Autocorrelation
    corr = np.correlate(audio, audio, mode="full")
    corr = corr[len(corr) // 2:]          # keep positive lags

    # Find first peak after lag 0 (the fundamental period)
    min_lag = int(sr / 500)   # max 500 Hz
    max_lag = int(sr / 50)    # min 50 Hz
    if max_lag >= len(corr):
        return 0.0

    search = corr[min_lag:max_lag]
    if len(search) == 0:
        return 0.0

    peak_lag  = np.argmax(search) + min_lag
    peak_val  = corr[peak_lag]
    noise_val = corr[0] - peak_val + 1e-8

    hnr_linear = peak_val / noise_val
    hnr_db     = 10 * np.log10(max(hnr_linear, 1e-8))
    return float(hnr_db)


def extract_spectral_features(audio: np.ndarray, sr: int,
                               n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extract spectral shape descriptors.
    Returns 8 values:
      centroid (mean, std), bandwidth (mean, std),
      rolloff  (mean, std), flatness  (mean, std)
    """
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=audio,  sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flatness  = librosa.feature.spectral_flatness(y=audio,         n_fft=n_fft, hop_length=hop_length)[0]

    return np.array([
        centroid.mean(),  centroid.std(),
        bandwidth.mean(), bandwidth.std(),
        rolloff.mean(),   rolloff.std(),
        flatness.mean(),  flatness.std(),
    ], dtype=np.float32)


def extract_zcr_features(audio: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """
    Zero Crossing Rate — how often the signal crosses zero.
    TTS voices often have unnaturally low or patterned ZCR.
    Returns 4 values: mean, std, min, max
    """
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    return np.array([zcr.mean(), zcr.std(), zcr.min(), zcr.max()], dtype=np.float32)


def extract_rms_features(audio: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """RMS energy stats — mean and std. Returns 2 values."""
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    return np.array([rms.mean(), rms.std()], dtype=np.float32)


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_voice_biometrics(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Full voice biometrics extraction pipeline.

    Returns a 22-dimensional float32 vector:
      [pitch×5, jitter×1, shimmer×1, hnr×1, spectral×8, zcr×4, rms×2]
    """
    pitch    = extract_pitch_features(audio, sr)          # 5
    jitter   = np.array([extract_jitter(audio, sr)],      dtype=np.float32)  # 1
    shimmer  = np.array([extract_shimmer(audio, sr)],     dtype=np.float32)  # 1
    hnr      = np.array([extract_hnr(audio, sr)],         dtype=np.float32)  # 1
    spectral = extract_spectral_features(audio, sr, n_fft, hop_length)       # 8
    zcr      = extract_zcr_features(audio, hop_length)    # 4
    rms      = extract_rms_features(audio, hop_length)    # 2

    features = np.concatenate([pitch, jitter, shimmer, hnr, spectral, zcr, rms])

    # Replace any NaN/Inf with 0 (can happen on very short/silent clips)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.astype(np.float32)


def extract_biometrics_from_file(
    path: Path,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Convenience wrapper: load file then extract biometrics."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return extract_voice_biometrics(audio, sr, n_fft, hop_length)