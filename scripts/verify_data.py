"""
Step 2c — Data Verification
=============================
After preprocessing, run this to confirm everything looks healthy
before moving to feature extraction.

Checks:
  - manifest.json exists and is valid
  - All files in manifest actually exist on disk
  - Class balance (real vs fake ratio)
  - Duration stats (min, max, mean)
  - Sample rate consistency
  - No corrupted files (tries to read each one)

Run:
    python scripts/verify_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger("verify")


def verify() -> None:
    manifest_path = ROOT / "data" / "processed" / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run: python scripts/preprocess.py first")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not manifest:
        logger.error("Manifest is empty!")
        sys.exit(1)

    logger.info(f"Manifest loaded: {len(manifest)} entries")

    # ── Check file existence ──────────────────────────────────────────────────
    missing   = []
    corrupted = []
    durations = []
    labels    = []
    sample_rates = Counter()

    for i, entry in enumerate(manifest):
        path = ROOT / entry["path"]
        labels.append(entry["label"])

        if not path.exists():
            missing.append(entry["path"])
            continue

        try:
            info = sf.info(str(path))
            durations.append(info.duration)
            sample_rates[info.samplerate] += 1
        except Exception as e:
            corrupted.append((entry["path"], str(e)))

        if (i + 1) % 100 == 0:
            print(f"  Verified {i+1}/{len(manifest)}...", end="\r")

    print()

    # ── Report ────────────────────────────────────────────────────────────────
    label_counts = Counter(labels)
    total        = len(manifest)
    real_count   = label_counts[0]
    fake_count   = label_counts[1]

    print("\n" + "=" * 55)
    print("  DATA VERIFICATION REPORT")
    print("=" * 55)

    print(f"\n  Total files      : {total}")
    print(f"  Real (label=0)   : {real_count}  ({100*real_count/total:.1f}%)")
    print(f"  Fake (label=1)   : {fake_count}  ({100*fake_count/total:.1f}%)")

    if durations:
        print(f"\n  Duration (sec)")
        print(f"    Min   : {min(durations):.2f}s")
        print(f"    Max   : {max(durations):.2f}s")
        print(f"    Mean  : {np.mean(durations):.2f}s")
        print(f"    Std   : {np.std(durations):.2f}s")

    print(f"\n  Sample rates     : {dict(sample_rates)}")
    print(f"  Missing files    : {len(missing)}")
    print(f"  Corrupted files  : {len(corrupted)}")

    if missing:
        print("\n  ❌ Missing:")
        for m in missing[:5]:
            print(f"     {m}")

    if corrupted:
        print("\n  ❌ Corrupted:")
        for path, err in corrupted[:5]:
            print(f"     {path}: {err}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "-" * 55)
    ok = not missing and not corrupted and len(sample_rates) == 1

    if ok:
        print("  ✅ All checks passed! Ready for Step 3: Feature Extraction")
    else:
        print("  ⚠️  Some issues found. Review above before continuing.")

    # Class imbalance warning
    ratio = max(real_count, fake_count) / max(min(real_count, fake_count), 1)
    if ratio > 2.0:
        print(f"  ⚠️  Class imbalance detected (ratio {ratio:.1f}x). "
              f"Consider oversampling or weighted loss.")

    print("=" * 55)


if __name__ == "__main__":
    verify()