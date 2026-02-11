import pandas as pd
from .dataloader.builder import dataloader_builder
from .histories.builder import histories_builder
from .interactions.interactions import interactions_generator
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


def pipeline_builder(
    df: pd.DataFrame, 
    cfg: dict,
    objective: str,
    seed: int,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    kwargs = dict(
        df=df,
        cfg=cfg["dataloader"],
        objective=objective,
        seed=seed,
        col_user=col_user,
        col_item=col_item,
    )
    dataloaders = dataloader_builder(**kwargs)

    df_trn = dataloaders["trn"].dataset.df

    kwargs = dict(
        df=df_trn,
        col_user=col_user,
        col_item=col_item,
    )
    interactions = interactions_generator(**kwargs)

    kwargs = dict(
        interactions=interactions,
        cfg=cfg["histories"],
    )
    histories = histories_builder(**kwargs)

    return dataloaders, interactions, histories