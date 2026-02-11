import torch.nn as nn
from .registry import OPTIMIZER_REGISTRY


def _base_optimizer(model, cfg):
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


def _grouped_optimizer(model, cfg):
    NAME = cfg["name"]
    BASE_LEARNING_RATE = cfg["lr"]
    BASE_WEIGHT_DECAY = cfg["weight_decay"]
    param_groups = cfg["param_groups"]

    opt_cls = OPTIMIZER_REGISTRY[NAME]

    groups = []

    handled_params = set()

    for module_name, module_cfg in param_groups.items():
        MODULE_LEARNING_RATE = (
            BASE_LEARNING_RATE * module_cfg.get("lr_scale", None)
            if module_cfg.get("lr_scale", None) is not None
            else BASE_LEARNING_RATE
        )
        MODULE_WEIGHT_DECAY = (
            module_cfg.get("weight_decay", None)
            if module_cfg.get("weight_decay", None) is not None 
            else BASE_WEIGHT_DECAY
        )
        
        module = getattr(model, module_name)
        module_params = list(module.parameters())

        kwargs = dict(
            params=module_params,
            lr=MODULE_LEARNING_RATE,
            weight_decay=MODULE_WEIGHT_DECAY,
        )
        groups.append(kwargs)

        for param in module_params:
            handled_params.add(id(param))

    remaining_params = [
        param
        for param in model.parameters()
        if id(param) not in handled_params
    ]

    if remaining_params:
        kwargs = dict(
            params=remaining_params,
            lr=BASE_LEARNING_RATE,
            weight_decay=BASE_WEIGHT_DECAY,
        )
        groups.append(kwargs)

    return opt_cls(groups)


def optimizer_builder(
    model: nn.Module,
    cfg: dict,
):
    return (
        _grouped_optimizer(model, cfg) 
        if "param_groups" in cfg.keys() 
        else _base_optimizer(model, cfg)
    )