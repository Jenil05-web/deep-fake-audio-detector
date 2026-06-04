"""
Step 3 — Feature Extraction Pipeline
======================================
Reads data/processed/manifest.json, runs all three extractors on every
file, and saves .npy arrays to data/features/.

Output structure:
  data/features/
    mfcc/
      real_0000.npy   shape: (120, 128)   — 3*n_mfcc × time_frames
      fake_0000.npy
      ...
    mel/
      real_0000.npy   shape: (1, 128, 128) — channel × freq × time
      fake_0000.npy
      ...
    biometrics/
      real_0000.npy   shape: (22,)         — 22-dim vector
      fake_0000.npy
      ...
    features_manifest.json  — maps each file to its .npy paths + label

Run:
    python scripts/extract_features.py
    python scripts/extract_features.py --limit 20    # quick test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import get_config as load_config
from src.utils.logger import get_logger
from src.features.mfcc import extract_mfcc
from src.features.mel_spectrogram import extract_mel_spectrogram
from src.features.voice_biometrics import extract_voice_biometrics

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install librosa soundfile")
    sys.exit(1)

logger = get_logger("extract_features")


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio


def run(limit: int | None = None) -> None:
    cfg = load_config()

    processed_dir = Path(cfg.paths.data_dir) / "processed"
    features_dir  = Path(cfg.paths.data_dir) / "features"
    manifest_path = processed_dir / "manifest.json"

    # ── Load manifest ──────────────────────────────────────────────────────────
    if not manifest_path.exists():
        logger.error(f"manifest.json not found at {manifest_path}")
        logger.error("Run Step 2 first: python scripts/preprocess.py")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    if limit:
        manifest = manifest[:limit]

    logger.info(f"Files to process: {len(manifest)}")

    # ── Create output dirs ─────────────────────────────────────────────────────
    mfcc_dir  = features_dir / "mfcc"
    mel_dir   = features_dir / "mel"
    bio_dir   = features_dir / "biometrics"
    for d in [mfcc_dir, mel_dir, bio_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Config values ──────────────────────────────────────────────────────────
    sr          = cfg.data.sample_rate
    n_mfcc      = cfg.data.n_mfcc
    n_fft       = cfg.data.n_fft
    hop_length  = cfg.data.hop_length
    fixed_len   = 128    # time frames — consistent across all features

    # ── Process each file ──────────────────────────────────────────────────────
    stats           = {"ok": 0, "error": 0}
    features_manifest = []
    t_start         = time.time()

    for i, entry in enumerate(manifest):
        audio_path = ROOT / entry["path"]

        if not audio_path.exists():
            logger.warning(f"File not found, skipping: {audio_path}")
            stats["error"] += 1
            continue

        # Derive output filenames from original audio stem
        stem      = Path(entry["path"]).stem
        mfcc_path = mfcc_dir / f"{stem}.npy"
        mel_path  = mel_dir  / f"{stem}.npy"
        bio_path  = bio_dir  / f"{stem}.npy"

        try:
            audio = load_audio(audio_path, sr)

            # ── Extract all three features ─────────────────────────────────────
            mfcc_feat = extract_mfcc(
                audio, sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
                fixed_length=fixed_len,
            )

            mel_feat = extract_mel_spectrogram(
                audio, sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=128,
                fixed_length=fixed_len,
            )

            bio_feat = extract_voice_biometrics(
                audio, sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            # ── Save ───────────────────────────────────────────────────────────
            np.save(str(mfcc_path), mfcc_feat)
            np.save(str(mel_path),  mel_feat)
            np.save(str(bio_path),  bio_feat)

            features_manifest.append({
                "audio_path":      entry["path"],
                "label":           entry["label"],
                "label_name":      entry["label_name"],
                "mfcc_path":       str(mfcc_path.resolve().relative_to(ROOT.resolve())),
                "mel_path":        str(mel_path.resolve().relative_to(ROOT.resolve())),
                "biometrics_path": str(bio_path.resolve().relative_to(ROOT.resolve())),
                "mfcc_shape":      list(mfcc_feat.shape),
                "mel_shape":       list(mel_feat.shape),
                "biometrics_dim":  int(bio_feat.shape[0]),
            })

            stats["ok"] += 1

        except Exception as e:
            logger.warning(f"Error on {audio_path.name}: {e}")
            stats["error"] += 1

        # Progress every 25 files
        if (i + 1) % 25 == 0 or (i + 1) == len(manifest):
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            logger.info(f"  [{i+1}/{len(manifest)}]  "
                        f"ok={stats['ok']}  error={stats['error']}  "
                        f"({rate:.1f} files/s)")

    # ── Save features manifest ─────────────────────────────────────────────────
    feat_manifest_path = features_dir / "features_manifest.json"
    with open(feat_manifest_path, "w") as f:
        json.dump(features_manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 55)
    logger.info(f"Done in {elapsed:.1f}s  |  ok={stats['ok']}  error={stats['error']}")
    logger.info(f"Feature shapes:")
    if features_manifest:
        ex = features_manifest[0]
        logger.info(f"  MFCC        : {ex['mfcc_shape']}   (3*n_mfcc × time_frames)")
        logger.info(f"  Mel         : {ex['mel_shape']}  (1 × n_mels × time_frames)")
        logger.info(f"  Biometrics  : ({ex['biometrics_dim']},)  (22-dim vector)")
    logger.info(f"Features manifest → {feat_manifest_path}")
    logger.info("Next: python scripts/verify_features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from preprocessed audio")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N files (quick test)")
    args = parser.parse_args()
    run(args.limit)