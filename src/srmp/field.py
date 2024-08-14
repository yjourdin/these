from dataclasses import dataclass

from numpy.random import Generator

from ..field import GeneratedField, group_generated_field
from .weight import balanced_weights, random_weights


@dataclass
class WeightsField(GeneratedField):
    weights: list[int]

    @classmethod
    def random(cls, nb_crit: int, rng: Generator, *args, **kwargs):
        super().random(*args, **kwargs)
        kwargs["weights"] = random_weights(nb_crit, rng)
        return kwargs["weights"]

    @classmethod
    def balanced(cls, nb_crit: int, *args, **kwargs):
        super().balanced(*args, **kwargs)
        kwargs["weights"] = balanced_weights(nb_crit)
        return kwargs["weights"]


# @dataclass
# class GroupWeightsField(
#     GeneratedField,
#     metaclass=GroupGeneratedField,
#     fieldname="weights",
#     fieldclass=WeightsField,
# ):
#     weights: list[list[int]]


@group_generated_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(GeneratedField):
    weights: list[list[int]]
