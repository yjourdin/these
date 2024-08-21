from dataclasses import dataclass

from numpy.random import Generator

from ..field import GeneratedField, generated_field, group_generated_field
from .weight import balanced_weights, random_weights


@generated_field("weights")
@dataclass
class WeightsField(GeneratedField):
    weights: list[float]

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
        return random_weights(nb_crit, rng)

    @staticmethod
    def field_balanced(nb_crit: int, *args, **kwargs):
        return balanced_weights(nb_crit)


@group_generated_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(GeneratedField):
    weights: list[list[float]]
