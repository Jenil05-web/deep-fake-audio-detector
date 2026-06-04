"""
Step 2a — Data Download Helper
================================
ASVspoof 2019 is the gold-standard dataset for fake audio detection.
It has 3 partitions: train, dev, eval — with real (bonafide) and fake (spoof) audio.

Download page: https://datashare.ed.ac.uk/handle/10283/3336
You need to register (free) and download manually on Windows.

This script:
  1. Tells you exactly what to download and where to put it
  2. Verifies the folder structure after you place the files
  3. Falls back to generating a small synthetic dataset for dev/testing
     so you can build the pipeline immediately without waiting for the download.

Run:
    python scripts/download_data.py --check     # verify your downloaded data
    python scripts/download_data.py --synthetic # generate fake dev data NOW
"""

import argparse
import os
import random
import struct
import wave
from pathlib import Path
import math

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner(msg: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def generate_sine_wav(path: Path, freq: float, duration: float, sr: int = 16000) -> None:
    """Write a simple sine-wave .wav — stands in for a 'real' voice."""
    n_samples = int(sr * duration)
    amplitude = 32767 * 0.3
    samples = [int(amplitude * math.sin(2 * math.pi * freq * t / sr))
               for t in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def generate_noise_wav(path: Path, duration: float, sr: int = 16000) -> None:
    """Write band-limited noise .wav — stands in for a 'fake' voice."""
    n_samples = int(sr * duration)
    amplitude = 32767 * 0.15
    samples = [int(random.gauss(0, amplitude)) for _ in range(n_samples)]
    samples = [max(-32767, min(32767, s)) for s in samples]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


# ── Commands ──────────────────────────────────────────────────────────────────

def print_download_instructions() -> None:
    print_banner("How to download ASVspoof 2019")
    print("""
STEP-BY-STEP (Windows):

1. Go to:
   https://datashare.ed.ac.uk/handle/10283/3336

2. Register for a free account and log in.

3. Download these two files (~12 GB total):
   - LA.zip   (Logical Access — TTS & voice conversion fakes)
   - PA.zip   (Physical Access — replay attacks)

   We only NEED LA.zip for this project (TTS/VC deepfakes).

4. Extract LA.zip. You'll get a folder like:
   LA/
     ASVspoof2019_LA_train/
       flac/          ← audio files
       ASVspoof2019.LA.cm.train.trn.txt  ← labels
     ASVspoof2019_LA_dev/
       flac/
       ASVspoof2019.LA.cm.dev.trl.txt
     ASVspoof2019_LA_eval/
       flac/
       ASVspoof2019.LA.cm.eval.trl.txt

5. Place the extracted LA/ folder here:
   data/raw/ASVspoof2019_LA/

6. Then run:  python scripts/download_data.py --check

ALTERNATIVE DATASETS (smaller, easier to start with):
  - WaveFake (4 GB):  https://github.com/RUB-SysSec/WaveFake
  - FakeAVCeleb:      https://github.com/DASH-Lab/FakeAVCeleb
    """)


def check_asvspooof() -> bool:
    print_banner("Checking ASVspoof 2019 LA folder structure")
    la_root = RAW / "ASVspoof2019_LA"
    expected = [
        la_root / "ASVspoof2019_LA_train" / "flac",
        la_root / "ASVspoof2019_LA_dev"   / "flac",
        la_root / "ASVspoof2019_LA_eval"  / "flac",
    ]
    all_ok = True
    for p in expected:
        exists = p.exists()
        status = "✅" if exists else "❌"
        count  = len(list(p.glob("*.flac"))) if exists else 0
        print(f"  {status} {p.relative_to(ROOT)}   ({count} files)")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n✅ Dataset looks good! Run Step 2b next: python scripts/preprocess.py")
    else:
        print("\n❌ Dataset not found. See instructions above, or run:")
        print("   python scripts/download_data.py --synthetic")
    return all_ok


def generate_synthetic(n_real: int = 150, n_fake: int = 150) -> None:
    print_banner(f"Generating synthetic dev dataset ({n_real} real + {n_fake} fake)")

    random.seed(42)
    real_dir = RAW / "real"
    fake_dir = RAW / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Real: sine waves at speech-like frequencies (100–300 Hz), 2–4 sec
    print(f"  Generating {n_real} real samples...")
    for i in range(n_real):
        freq     = random.uniform(100, 300)
        duration = random.uniform(2.0, 4.0)
        generate_sine_wav(real_dir / f"real_{i:04d}.wav", freq, duration)

    # Fake: Gaussian noise bursts (simulate vocoder artifacts), 2–4 sec
    print(f"  Generating {n_fake} fake samples...")
    for i in range(n_fake):
        duration = random.uniform(2.0, 4.0)
        generate_noise_wav(fake_dir / f"fake_{i:04d}.wav", duration)

    total = n_real + n_fake
    print(f"\n✅ Synthetic dataset ready: {total} files in data/raw/")
    print("   real/ →", n_real, "files")
    print("   fake/ →", n_fake, "files")
    print("\nNext: python scripts/preprocess.py")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data download helper")
    parser.add_argument("--check",     action="store_true", help="Check ASVspoof folder")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dev data")
    parser.add_argument("--n-real",    type=int, default=150)
    parser.add_argument("--n-fake",    type=int, default=150)
    args = parser.parse_args()

    if args.check:
        check_asvspooof()
    elif args.synthetic:
        generate_synthetic(args.n_real, args.n_fake)
    else:
        print_download_instructions()
        print("\nTIP: Run with --synthetic to generate test data immediately.")