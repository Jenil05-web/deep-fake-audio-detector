"""
Step 4 — PyTorch Dataset
==========================
Loads pre-extracted .npy feature files and serves them to the
DataLoader during training.

Returns a dict with all three feature types so each model
(CNN, LSTM, Ensemble) can pick what it needs:

    {
        "mel":        Tensor (1, 128, 128)  ← for CNN
        "mfcc":       Tensor (120, 128)     ← for LSTM  (120 = 3*n_mfcc)
        "biometrics": Tensor (22,)          ← for ensemble MLP head
        "label":      Tensor scalar (0 or 1)
    }

Usage:
    from src.training.dataset import DeepfakeDataset, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).parent.parent.parent


class DeepfakeDataset(Dataset):
    """
    Loads MFCC / Mel / Biometric .npy files from features_manifest.json.

    Args:
        entries:   List of dicts from features_manifest.json
        augment:   If True, apply lightweight augmentation (training only)
    """

    def __init__(self, entries: list, augment: bool = False):
        self.entries = entries
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]

        # ── Load .npy arrays ──────────────────────────────────────────────────
        mel        = np.load(ROOT / entry["mel_path"])          # (1, 128, 128)
        mfcc       = np.load(ROOT / entry["mfcc_path"])         # (120, 128)
        biometrics = np.load(ROOT / entry["biometrics_path"])   # (22,)
        label      = int(entry["label"])                        # 0=real, 1=fake

        # ── Optional augmentation (training only) ─────────────────────────────
        if self.augment:
            mel, mfcc = self._augment(mel, mfcc)

        return {
            "mel":        torch.tensor(mel,        dtype=torch.float32),
            "mfcc":       torch.tensor(mfcc,       dtype=torch.float32),
            "biometrics": torch.tensor(biometrics, dtype=torch.float32),
            "label":      torch.tensor(label,      dtype=torch.long),
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(
        self,
        mel: np.ndarray,
        mfcc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lightweight spectrogram augmentation.
        Applied in numpy before converting to tensor (fast, no extra deps).
        """
        # 1. Gaussian noise
        if np.random.rand() < 0.5:
            mel  = mel  + np.random.normal(0, 0.01, mel.shape).astype(np.float32)
            mfcc = mfcc + np.random.normal(0, 0.01, mfcc.shape).astype(np.float32)

        # 2. Time masking — zero out up to 15% of time frames
        if np.random.rand() < 0.5:
            # FIXED: Force integer conversion and use [-1] to dynamically grab the last dimension
            T          = int(mel.shape[-1])
            mask_len   = np.random.randint(1, max(2, int(T * 0.15)))
            mask_start = np.random.randint(0, max(1, T - mask_len))
            
            # FIXED: Using ellipses (...) to safely slice regardless of array depth
            mel[..., mask_start:mask_start + mask_len]  = 0
            mfcc[..., mask_start:mask_start + mask_len] = 0

        # 3. Frequency masking — zero out up to 15% of frequency bins
        if np.random.rand() < 0.5:
            # FIXED: Force integer conversion for frequency dimension
            F          = int(mel.shape[-2])
            mask_len   = np.random.randint(1, max(2, int(F * 0.15)))
            mask_start = np.random.randint(0, max(1, F - mask_len))
            mel[..., mask_start:mask_start + mask_len, :] = 0

        return mel, mfcc


# ── Split + DataLoader factory ────────────────────────────────────────────────

def load_manifest(features_dir: Path) -> list:
    """Load features_manifest.json from the features directory."""
    manifest_path = features_dir / "features_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"features_manifest.json not found at {manifest_path}\n"
            "Run Step 3 first: python scripts/extract_features.py"
        )
    with open(manifest_path) as f:
        return json.load(f)


def split_manifest(
    entries: list,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """
    Stratified split into train / val / test.
    Stratified = same real/fake ratio in every split.
    """
    rng = np.random.default_rng(seed)

    real = [e for e in entries if e["label"] == 0]
    fake = [e for e in entries if e["label"] == 1]

    def split_class(items):
        items = list(items)
        rng.shuffle(items)
        n       = len(items)
        n_test  = max(1, int(n * test_split))
        n_val   = max(1, int(n * val_split))
        test    = items[:n_test]
        val     = items[n_test:n_test + n_val]
        train   = items[n_test + n_val:]
        return train, val, test

    real_train, real_val, real_test = split_class(real)
    fake_train, fake_val, fake_test = split_class(fake)

    train = real_train + fake_train
    val   = real_val   + fake_val
    test  = real_test  + fake_test

    # Shuffle train so batches aren't all-real then all-fake
    rng.shuffle(train)

    return train, val, test


def make_weighted_sampler(entries: list) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that balances class frequency.
    """
    labels       = [e["label"] for e in entries]
    class_counts = np.bincount(labels)
    weights      = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Full pipeline: load manifest → split → Dataset → DataLoader."""
    features_dir = Path(cfg.paths.data_dir) / "features"
    entries      = load_manifest(features_dir)

    train_entries, val_entries, test_entries = split_manifest(
        entries,
        val_split=cfg.training.validation_split,
        test_split=0.15,
        seed=42,
    )

    train_ds = DeepfakeDataset(train_entries, augment=True)
    val_ds   = DeepfakeDataset(val_entries,   augment=False)
    test_ds  = DeepfakeDataset(test_entries,  augment=False)

    sampler = make_weighted_sampler(train_entries)
    num_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def get_dataset_stats(cfg) -> dict:
    """Print a summary of the dataset splits."""
    features_dir = Path(cfg.paths.data_dir) / "features"
    entries       = load_manifest(features_dir)
    train, val, test = split_manifest(
        entries,
        val_split=cfg.training.validation_split,
        test_split=0.15,
        seed=42,
    )

    def count(split):
        real = sum(1 for e in split if e["label"] == 0)
        fake = sum(1 for e in split if e["label"] == 1)
        return {"total": len(split), "real": real, "fake": fake}

    return {
        "total":    len(entries),
        "train":    count(train),
        "val":      count(val),
        "test":     count(test),
    }