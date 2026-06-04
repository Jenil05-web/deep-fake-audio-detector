"""
Step 4b — Verify Dataset
==========================
Confirms the Dataset and DataLoader work correctly before training.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config as get_config
from src.utils.logger import get_logger
from src.training.dataset import (
    DeepfakeDataset,
    get_dataloaders,
    get_dataset_stats,
    load_manifest,
    split_manifest,
)

logger = get_logger("verify_dataset")

def verify() -> None:
    cfg = get_config()

    # ── 1. Stats ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DATASET VERIFICATION")
    print("=" * 55)

    try:
        stats = get_dataset_stats(cfg)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    print(f"\n  Total samples : {stats['total']}")
    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        print(f"  {split_name:<6}        : {s['total']:>4} total  "
              f"| real={s['real']}  fake={s['fake']}")

    # ── 2. Single sample check ────────────────────────────────────────────────
    print("\n  Checking single sample shapes...")
    
    features_dir = Path(cfg.paths.data_dir) / "features"
    entries      = load_manifest(features_dir)
    train_e, _, _= split_manifest(entries)

    if not train_e:
        logger.error("No training entries found!")
        sys.exit(1)

    # =========================================================================
    # BULLETPROOF FETCH: Using explicit __getitem__(0) instead of brackets 
    # to guarantee we extract the dictionary, not the dataset itself.
    # =========================================================================
    ds = DeepfakeDataset(train_e[:5], augment=False)
    
    # We bypass brackets completely so Python can't get confused
    sample_dict = ds.__getitem__(0) 

    print(f"    mel        : {tuple(sample_dict['mel'].shape)}  expected (1, 128, 128)")
    print(f"    mfcc       : {tuple(sample_dict['mfcc'].shape)}  expected (120, 128)")
    print(f"    biometrics : {tuple(sample_dict['biometrics'].shape)}    expected (22,)")
    print(f"    label      : {sample_dict['label'].item()}  (0=real, 1=fake)")
    print(f"    dtypes     : mel={sample_dict['mel'].dtype}  "
          f"mfcc={sample_dict['mfcc'].dtype}  label={sample_dict['label'].dtype}")

    assert sample_dict["mel"].shape        == (1, 128, 128), "Mel shape mismatch!"
    assert sample_dict["mfcc"].shape       == (120, 128),    "MFCC shape mismatch!"
    assert sample_dict["biometrics"].shape == (22,),         "Biometrics shape mismatch!"

    # ── 3. Augmentation check ─────────────────────────────────────────────────
    print("\n  Checking augmentation...")
    ds_aug = DeepfakeDataset(train_e[:5], augment=True)
    
    for i in range(3):
        # Using explicit fetch here too just to be safe
        s = ds_aug.__getitem__(i)
        assert s["mel"].shape  == (1, 128, 128)
        assert s["mfcc"].shape == (120, 128)
        assert torch.isfinite(s["mel"]).all(),  "NaN in augmented mel!"
        assert torch.isfinite(s["mfcc"]).all(), "NaN in augmented mfcc!"
    print("    ✅ Augmentation OK")

    # ── 4. DataLoader batch check ─────────────────────────────────────────────
    print("\n  Checking DataLoader batches...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Using next(iter()) is completely safe for loaders
    batch = next(iter(train_loader))
    bs    = batch["mel"].shape
    
    print(f"    batch_size : {bs}")
    print(f"    mel batch  : {tuple(batch['mel'].shape)}")
    print(f"    mfcc batch : {tuple(batch['mfcc'].shape)}")
    print(f"    bio batch  : {tuple(batch['biometrics'].shape)}")
    print(f"    labels     : {batch['label'].tolist()}")

    print(f"\n  Batches per epoch:")
    print(f"    train : {len(train_loader)}")
    print(f"    val   : {len(val_loader)}")
    print(f"    test  : {len(test_loader)}")

    print("\n" + "-" * 55)
    print("  ✅ All checks passed! Ready for Step 5: Model Training")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    verify()