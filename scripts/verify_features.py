"""
Step 3b — Feature Verification
================================
Sanity-checks extracted .npy feature files before training.

Checks:
  - features_manifest.json exists
  - All .npy files exist and are loadable
  - Shape consistency across all samples
  - No NaN / Inf values
  - Class balance
  - Prints sample statistics (mean, std, min, max per feature type)

Run:
    python scripts/verify_features.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger("verify_features")


def verify() -> None:
    feat_dir      = ROOT / "data" / "features"
    manifest_path = feat_dir / "features_manifest.json"

    if not manifest_path.exists():
        logger.error("features_manifest.json not found.")
        logger.error("Run: python scripts/extract_features.py")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info(f"Manifest entries: {len(manifest)}")

    issues        = []
    label_counts  = Counter()
    mfcc_shapes   = Counter()
    mel_shapes    = Counter()
    bio_dims      = Counter()
    nan_counts    = {"mfcc": 0, "mel": 0, "bio": 0}

    # Collect stats for summary
    bio_samples = []

    for entry in manifest:
        label_counts[entry["label_name"]] += 1

        for key, feat_type in [
            ("mfcc_path", "mfcc"),
            ("mel_path",  "mel"),
            ("biometrics_path", "bio"),
        ]:
            path = ROOT / entry[key]
            if not path.exists():
                issues.append(f"Missing: {path}")
                continue

            try:
                arr = np.load(str(path))
            except Exception as e:
                issues.append(f"Load error {path.name}: {e}")
                continue

            # Shape tracking
            if feat_type == "mfcc":
                mfcc_shapes[str(arr.shape)] += 1
            elif feat_type == "mel":
                mel_shapes[str(arr.shape)] += 1
            else:
                bio_dims[arr.shape[0]] += 1
                bio_samples.append(arr)

            # NaN / Inf check
            if not np.isfinite(arr).all():
                nan_counts[feat_type] += 1
                issues.append(f"NaN/Inf in {feat_type}: {path.name}")

    # ── Print report ───────────────────────────────────────────────────────────
    total = len(manifest)
    print("\n" + "=" * 58)
    print("  FEATURE VERIFICATION REPORT")
    print("=" * 58)

    print(f"\n  Total samples   : {total}")
    for lname, count in label_counts.items():
        print(f"  {lname:<14}  : {count}  ({100*count/max(total,1):.1f}%)")

    print(f"\n  MFCC shapes     : {dict(mfcc_shapes)}")
    print(f"  Mel shapes      : {dict(mel_shapes)}")
    print(f"  Biometric dims  : {dict(bio_dims)}")

    print(f"\n  NaN/Inf counts  : mfcc={nan_counts['mfcc']}  "
          f"mel={nan_counts['mel']}  bio={nan_counts['bio']}")
    print(f"  Other issues    : {len(issues)}")

    if bio_samples:
        bio_arr = np.stack(bio_samples)
        print(f"\n  Biometric stats (across {len(bio_samples)} samples):")
        print(f"    mean : {bio_arr.mean(axis=0).round(4)}")
        print(f"    std  : {bio_arr.std(axis=0).round(4)}")
        print(f"    min  : {bio_arr.min(axis=0).round(4)}")
        print(f"    max  : {bio_arr.max(axis=0).round(4)}")

    print("\n" + "-" * 58)
    if not issues:
        print("  ✅ All checks passed! Ready for Step 4: Dataset Class")
    else:
        print(f"  ⚠️  {len(issues)} issues found:")
        for iss in issues[:10]:
            print(f"     {iss}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    verify()