from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from ..field import (
    RandomField,
    random_field,
    random_group_field,
)
from .weight import random_weights


@random_field("weights")
@dataclass
class WeightsField(RandomField):
    weights: np.ndarray

    @staticmethod
    def field_decode(o):
        return np.array(o)

    @staticmethod
    def field_encode(o):
        return o.tolist()

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
        return random_weights(nb_crit, rng)


@random_group_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(RandomField):
    weights: list[np.ndarray]
