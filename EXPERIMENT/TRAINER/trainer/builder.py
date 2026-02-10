import torch
import torch.nn as nn
from .registry import TRAINER_REGISTRY
from ..loss_fn.registry import CRITERION_REGISTRY


def builder(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: str,
    objective: str,
):
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=CRITERION_REGISTRY[objective][criterion],
    )
    return TRAINER_REGISTRY[objective](**kwargs)
