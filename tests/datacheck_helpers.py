"""数据检查测试共享辅助。"""

import os
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def _doc(i):
    return f"Document {i}: " + " ".join(f"word{j}" for j in range(20)) + "."


def _write_min_splits(d: str) -> None:
    with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
        f.write("alpha beta gamma delta epsilon zeta eta theta iota kappa.\n")
    with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
        f.write("val text with enough characters to survive sanitization.\n")
    with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
        f.write("test text with enough characters to survive sanitization.\n")
