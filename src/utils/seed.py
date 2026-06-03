"""Reproducibility helper - Set seeds for all randomization sources."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For reproducibility on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
