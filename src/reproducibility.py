"""
Set all random seeds from one place.
"""

import logging
import os
import random

import numpy as np

from .config import RANDOM_SEED

logger = logging.getLogger(__name__)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).
    Parameters
    ----------
    seed : int
        Seed value.  Defaults to RANDOM_SEED from config (42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed} (Python / NumPy / PyTorch)")