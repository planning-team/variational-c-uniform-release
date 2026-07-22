import os
import random
import numpy as np
import torch


def seed_all(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA GPUs) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_task_seeds(master_seed: int, n_tasks: int) -> list[int]:
    """Generate deterministic, statistically independent seeds for parallel tasks.

    Uses NumPy's SeedSequence to produce high-quality independent streams
    regardless of how tasks are distributed across workers.
    """
    ss = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n_tasks)]
