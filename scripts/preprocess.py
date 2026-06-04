"""
Step 2b — Audio Preprocessing Pipeline
========================================
Takes raw audio (ASVspoof .flac OR synthetic .wav) and produces
clean, normalized, fixed-length .wav files in data/processed/.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config as get_config
from src.utils.logger import get_logger

logger = get_logger("preprocess")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not installed — using soundfile fallback (no resampling)")


# ── Core audio functions ───────────────────────────────────────────────────────

def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr."""
    if HAS_LIBROSA:
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    else:
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        audio = audio.astype(np.float32)
    return audio, sr


def apply_vad(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Voice Activity Detection: trim silence from start and end."""
    # SAFE CHECK
    if audio is None or len(audio) == 0:
        return audio
        
    if HAS_LIBROSA:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    else:
        # Simple energy-based fallback
        frame_len = int(sr * 0.02)  # 20ms frames
        if len(audio) < frame_len:
            return audio
            
        energy = np.array([
            np.sqrt(np.mean(audio[i:i+frame_len]**2))
            for i in range(0, len(audio) - frame_len, frame_len)
        ])
        threshold = energy.max() * 0.01
        active = np.where(energy > threshold)
        
        # SAFE CHECK
        if len(active) == 0:
            return audio
            
        start = active * frame_len
        end   = min((active[-1] + 1) * frame_len, len(audio))
        return audio[start:end]


def clip_or_pad(audio: np.ndarray, sr: int, max_sec: float, min_sec: float) -> np.ndarray | None:
    """Clip to max_sec or pad with zeros to min_sec."""
    # SAFE CHECK
    if audio is None or len(audio) == 0:
        return None
        
    max_samples = int(sr * max_sec)
    min_samples = int(sr * min_sec)

    if len(audio) < min_samples:
        repeats = int(np.ceil(min_samples / len(audio)))
        audio = np.tile(audio, repeats)[:min_samples]

    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude of 1.0."""
    # SAFE CHECK
    if audio is None or len(audio) == 0:
        return audio
        
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio  # avoid division by near-zero
    return audio / peak


def process_file(
    src_path: Path,
    dst_path: Path,
    target_sr: int,
    max_sec: float,
    min_sec: float,
) -> dict:
    """Full preprocessing pipeline for one file."""
    try:
        audio, sr = load_audio(src_path, target_sr)
        original_duration = len(audio) / sr

        audio = apply_vad(audio, sr)
        audio = clip_or_pad(audio, sr, max_sec, min_sec)
        
        # SAFE CHECK
        if audio is None or len(audio) == 0:
            return {"status": "skipped", "reason": "too short after VAD"}

        audio = peak_normalize(audio)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), audio, target_sr, subtype="PCM_16")

        return {
            "status": "ok",
            "original_duration": round(original_duration, 2),
            "processed_duration": round(len(audio) / target_sr, 2),
            "sample_rate": target_sr,
        }

    except Exception as e:
        return {"status": "error", "reason": traceback.format_exc()}


# ── Dataset adapters ──────────────────────────────────────────────────────────

def iter_synthetic(raw_dir: Path):
    for f in sorted((raw_dir / "real").glob("*.wav")):
        yield f, 0
    for f in sorted((raw_dir / "fake").glob("*.wav")):
        yield f, 1


def iter_asvspooof(raw_dir: Path):
    la_root = raw_dir / "ASVspoof2019_LA"
    label_files = {
        "train": la_root / "ASVspoof2019_LA_train" / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   la_root / "ASVspoof2019_LA_dev"   / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  la_root / "ASVspoof2019_LA_eval"  / "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    audio_dirs = {
        "train": la_root / "ASVspoof2019_LA_train" / "flac",
        "dev":   la_root / "ASVspoof2019_LA_dev"   / "flac",
        "eval":  la_root / "ASVspoof2019_LA_eval"  / "flac",
    }
    for split, lf in label_files.items():
        if not lf.exists():
            continue
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id  = parts
                label    = 0 if parts == "bonafide" else 1
                audio_path = audio_dirs[split] / f"{file_id}.flac"
                if audio_path.exists():
                    yield audio_path, label


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(source: str = "synthetic", limit: int | None = None) -> None:
    cfg = get_config()
    
    # FIX: Use data_dir to build the raw and processed paths dynamically
    base_data_dir = Path(cfg.paths.data_dir)
    raw_dir = base_data_dir / "raw"
    out_dir = base_data_dir / "processed"

    logger.info(f"Source: {source} | Raw: {raw_dir} | Output: {out_dir}")
    
    # FIX: Point to cfg.data instead of cfg.audio
    logger.info(f"Target SR: {cfg.data.sample_rate} Hz | "
                f"Max: {cfg.data.duration}s | "
                f"Min: 1s")

    if source == "asvspooof":
        items = list(iter_asvspooof(raw_dir))
    else:
        items = list(iter_synthetic(raw_dir))

    if not items:
        logger.error("No audio files found! Run: python scripts/download_data.py --synthetic")
        sys.exit(1)

    if limit:
        items = items[:limit]

    logger.info(f"Found {len(items)} files to process")

    stats   = {"ok": 0, "skipped": 0, "error": 0}
    manifest = [] 

    for i, (src_path, label) in enumerate(items):
        label_name = "real" if label == 0 else "fake"
        dst_path   = out_dir / label_name / src_path.with_suffix(".wav").name

        result = process_file(
            src_path, dst_path,
            # FIX: Use matching config names
            target_sr=cfg.data.sample_rate,
            max_sec=cfg.data.duration,
            min_sec=1,
        )

        stats[result["status"]] = stats.get(result["status"], 0) + 1

        if result["status"] == "error":
            logger.error(f"CRASH on {src_path.name}:\n{result['reason']}")
            sys.exit(1)

        if result["status"] == "ok":
            manifest.append({
                "path": str(dst_path.resolve().relative_to(ROOT.resolve())),
                "label": label,
                "label_name": label_name,
                **result,
            })

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            logger.info(f"  [{i+1}/{len(items)}] ok={stats['ok']} "
                        f"skipped={stats['skipped']} error={stats['error']}")

    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 50)
    logger.info(f"Done! {stats['ok']} processed | "
                f"{stats['skipped']} skipped | {stats['error']} errors")
    logger.info(f"Manifest saved → {manifest_path}")
    logger.info("Next step: python scripts/verify_data.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("--source", choices=["synthetic", "asvspooof"],
                        default="synthetic", help="Which dataset to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N files (for testing)")
    args = parser.parse_args()
    run(args.source, args.limit)