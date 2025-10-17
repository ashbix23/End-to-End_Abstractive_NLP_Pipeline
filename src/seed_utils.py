import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.
    Works on CPU-only machines (macOS included).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        # Apple Metal backend — still deterministic behavior
        torch.mps.manual_seed(seed)

    # Disable nondeterministic optimizations
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

