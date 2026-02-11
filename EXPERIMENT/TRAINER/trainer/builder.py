import torch.nn as nn
from .objective.registry import TRAINER_REGISTRY
from .loss_fn.registry import CRITERION_REGISTRY
from .optimizer.factory import optimizer_factory


def builder(
    model: nn.Module,
    cfg: dict,
):
    OBJECTIVE = cfg["objective"]
    CRITERION = cfg["trainer"]["criterion"]

    kwargs = dict(
        model=model,
        cfg=cfg["optimizer"],
    )
    optimizer = optimizer_factory(**kwargs)
    
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=CRITERION_REGISTRY[OBJECTIVE][CRITERION],
    )
    return TRAINER_REGISTRY[OBJECTIVE](**kwargs)
