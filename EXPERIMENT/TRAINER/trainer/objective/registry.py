from .pointwise import PointwiseTrainer
from .pairwise import PairwiseTrainer
from .listwise import ListwiseTrainer


TRAINER_REGISTRY = {
    "pointwise": PointwiseTrainer,
    "pairwise": PairwiseTrainer,
    "listwise": ListwiseTrainer,
}