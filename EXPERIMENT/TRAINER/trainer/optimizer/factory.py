import torch.nn as nn
from .registry import OPTIMIZER_REGISTRY


def optimizer_factory(
    model: nn.Module,
    cfg: dict,
):
    NAME = cfg["name"]
    LEARNING_RATE = cfg["lr"]
    WEIGHT_DECAY = cfg["weight_decay"]

    opt_cls = OPTIMIZER_REGISTRY[NAME]

    kwargs = dict(
        params=model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
    )
    return opt_cls(**kwargs)