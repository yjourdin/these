from dataclasses import dataclass
from typing import Any

from numpy.random import Generator

from ..field import GeneratedField, group_generated_field
from .weight import balanced_weights, random_weights


@dataclass
class WeightsField(GeneratedField):
    weights: list[int]

    @classmethod
    def random(
        cls,
        nb_crit: int,
        rng: Generator,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs
    ):
        super().random(nb_crit=nb_crit, rng=rng, init_dict=init_dict, *args, **kwargs)
        init_dict["weights"] = random_weights(nb_crit, rng)
        return init_dict["weights"]

    @classmethod
    def balanced(cls, nb_crit: int, init_dict: dict[str, Any] = {}, *args, **kwargs):
        super().balanced(nb_crit=nb_crit, init_dict=init_dict, *args, **kwargs)
        init_dict["weights"] = balanced_weights(nb_crit)
        return init_dict["weights"]


@group_generated_field(fieldname="weights", fieldclass=WeightsField)
@dataclass
class GroupWeightsField(GeneratedField):
    weights: list[list[int]]
