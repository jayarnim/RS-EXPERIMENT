import copy
import torch
import torch.nn as nn
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from .early_stopper import EarlyStopper
from .predictor import Predictor
from .metrics import MetricsComputer
from ..metric_fn.registry import CRITERION_REGISTRY


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monitor:
    def __init__(
        self,
        model: nn.Module,
        criterion: str,
        delta: float,
        patience: int,
        warmup: int,
        k: int=DEFAULT_K,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        # global attr
        self.model = model.to(DEVICE)
        self.criterion = CRITERION_REGISTRY[criterion]
        self.delta = delta
        self.patience = patience
        self.warmup = warmup
        self.k = k
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        self._set_up_components()

    def __call__(
        self,
        loo_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        num_epochs: int,
    ):
        kwargs = dict(
            loo_loader=loo_loader,
            current_epoch=epoch,
            num_epochs=num_epochs,
        )
        result = self.predictor(**kwargs)
        
        score = self.metrics(result)

        kwargs = dict(
            current_score=score, 
            current_epoch=epoch,
            current_model_state=copy.deepcopy(self.model.state_dict()),
        )
        self.stopper(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.stopper.should_stop

    @property
    def counter(self):
        return self.stopper.counter

    @property
    def best_epoch(self):
        return self.stopper.best_epoch

    @property
    def best_score(self):
        return self.stopper.best_score

    @property
    def best_model_state(self):
        return self.stopper.best_model_state

    def _set_up_components(self):
        self._init_predictor()
        self._init_metrics()
        self._init_stopper()

    def _init_predictor(self):
        kwargs = dict(
            model=self.model,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        self.predictor = Predictor(**kwargs)

    def _init_metrics(self):
        kwargs = dict(
            criterion=self.criterion,
            k=self.k,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        self.metrics = MetricsComputer(**kwargs)

    def _init_stopper(self):        
        kwargs = dict(
            delta=self.delta,
            patience=self.patience,
            warmup=self.warmup,
        )
        self.stopper = EarlyStopper(**kwargs)