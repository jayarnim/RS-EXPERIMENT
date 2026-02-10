import torch
from torch.nn.utils.rnn import pad_sequence
from .selector.registry import SELECTOR_REGISTRY


def builder(
    interactions: torch.Tensor, 
    max_hist: int=None, 
    selector: str="default",
):
    # drop padding idx
    interactions_unpadded = interactions[:-1, :-1]

    # padding idx
    n_target, n_counterpart = interactions_unpadded.shape

    # select hist per target
    kwargs = dict(
        interactions=interactions_unpadded,
        max_hist=max_hist,
    )
    hist_indices = SELECTOR_REGISTRY[selector](**kwargs)

    # padding
    kwargs = dict(
        sequences=hist_indices, 
        batch_first=True, 
        padding_value=n_counterpart,
    )
    hist_indices_padded = pad_sequence(**kwargs)

    return hist_indices_padded