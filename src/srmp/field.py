from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..field import (
    RandomField,
    random_field,
    random_group_field,
)
from ..utils import tolist
from .weight import frozen_importance_relation_from_weights, random_weights


@random_field("weights")
@dataclass
class WeightsField(RandomField):
    weights: npt.NDArray[np.float64]

    @staticmethod
    def field_decode(o: Any):
        return np.array(o)

    @staticmethod
    def field_encode(o: Any):
        return tolist(o)

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args: Any, **kwargs: Any):
        return random_weights(nb_crit, rng)


@random_field("weights")
@dataclass(frozen=True)
class FrozenWeightsField(RandomField):
    weights: npt.NDArray[np.float64] = field(compare=False)
    # weights: tuple[float, ...]
    importance_relation: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "importance_relation",
            frozen_importance_relation_from_weights(self.weights),
        )

    @staticmethod
    def field_decode(o: Any):
        return np.array(o)

    @staticmethod
    def field_encode(o: Any):
        return list(o)


@random_group_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(RandomField):
    weights: list[npt.NDArray[np.float64]]
