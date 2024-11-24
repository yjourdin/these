from dataclasses import dataclass

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
    weights: list[float]

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
        return random_weights(nb_crit, rng)


@random_group_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(RandomField):
    weights: list[list[float]]
