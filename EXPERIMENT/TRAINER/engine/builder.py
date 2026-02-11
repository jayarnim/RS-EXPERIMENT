import torch.nn as nn
import torch.optim as optim
from .engine.registry import ENGINE_REGISTRY
from .loss_fn.registry import CRITERION_REGISTRY


def engine_builder(
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: dict,
    objective: str,
):
    CRITERION = cfg["criterion"]

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=CRITERION_REGISTRY[objective][CRITERION],
    )
    return ENGINE_REGISTRY[objective](**kwargs)
