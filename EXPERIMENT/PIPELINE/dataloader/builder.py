import pandas as pd
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from ...msr.python_splitters import python_stratified_split
from .objective.registry import DATALOADER_REGISTRY


def _data_stratified_splitter(df, ratio_trn_val_tst, seed, col_user, col_item):
    # for leave one out data set
    loo = (
        df
        .groupby(col_user)
        .sample(n=1, random_state=seed)
        .sort_values(by=col_user)
        .reset_index(drop=True)
    )

    # for trn, val, tst data set
    trn_val_tst = (
        df[~df[[col_user, col_item]]
        .apply(tuple, axis=1)
        .isin(set(loo[[col_user, col_item]]
        .apply(tuple, axis=1)))]
        .reset_index(drop=True)
    )

    # trn_val_tst -> [trn, val, tst]
    kwargs = dict(
        data=trn_val_tst,
        ratio=ratio_trn_val_tst,
        col_user=col_user,
        col_item=col_item,
        seed=seed,
    )
    split_list = python_stratified_split(**kwargs)
    
    split_list.append(loo)

    return split_list

def _candidates_generator(df, col_user, col_item):
    user_list = sorted(df[col_user].unique())
    item_list = sorted(df[col_item].unique())

    pos_per_user = {
        user: set(df.loc[df[col_user]==user, col_item].tolist())
        for user in user_list
    }

    neg_per_user = {
        user: list(set(item_list) - pos_per_user[user])
        for user in user_list
    }

    return neg_per_user

def _dataloader_generator(objective, split_list, candidates, ratio_neg_per_pos, batch_size, shuffle):
    dataloader_list = []

    for i, split in enumerate(split_list):
        kwargs = dict(
            df=split, 
            candidates=candidates,
            ratio_neg_per_pos=ratio_neg_per_pos if i<2 else 99, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        dataloader = (
            DATALOADER_REGISTRY[objective](**kwargs)
            if i<2 
            else DATALOADER_REGISTRY["pointwise"](**kwargs)
        )
        dataloader_list.append(dataloader)

    return dataloader_list


def builder(
    df: pd.DataFrame,
    objective: str,
    ratio_trn_val_tst: list=[8, 1, 1],
    ratio_neg_per_pos: int=4,
    batch_size: int=128,
    shuffle: bool=True,
    seed: int=42,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    # split original data
    kwargs = dict(
        df=df,
        ratio_trn_val_tst=ratio_trn_val_tst,
        seed=seed,
        col_user=col_user,
        col_item=col_item,
    )
    split_list = _data_stratified_splitter(**kwargs)

    kwargs = dict(
        df=df,
        col_user=col_user,
        col_item=col_item,
    )
    candidates = _candidates_generator(**kwargs)

    # generate data loaders
    kwargs = dict(
        objective=objective,
        split_list=split_list, 
        candidates=candidates,
        ratio_neg_per_pos=ratio_neg_per_pos, 
        batch_size=batch_size, 
        shuffle=shuffle,
    )
    dataloader_list = _dataloader_generator(**kwargs)

    return dataloader_list