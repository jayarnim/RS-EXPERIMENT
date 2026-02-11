import random
import numpy as np
import torch


def reset(
    cfg: dict,
):
    SEED = cfg["seed"]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"ALL SEEDS RESET: {SEED}")
