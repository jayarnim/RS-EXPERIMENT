import torch
import torch.nn as nn
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)
from .predictor import evaluation_predictor
from .metrics import metrics_computer


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def performance_evaluator(
    model: nn.Module, 
    tst_loader: torch.utils.data.dataloader.DataLoader,
    top_k_list: list=[5, 10, 15, 20, 25, 50, 100],
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    kwargs = dict(
        model=model,
        tst_loader=tst_loader,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    result = evaluation_predictor(**kwargs)

    kwargs = dict(
        result=result,
        top_k_list=top_k_list,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    metrics_sheet = metrics_computer(**kwargs)

    return metrics_sheet