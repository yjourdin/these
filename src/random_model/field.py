from dataclasses import dataclass, field
from typing import Any

from numpy.random import Generator

from ..field import RandomField, random_field
from ..random import Seed, seed


@random_field("seed")
@dataclass
class SeedField(RandomField):
    seed: Seed = field(default=seed())

    @staticmethod
    def field_random(
        rng: Generator,
        *args: Any,
        **kwargs: Any,
    ):
        return seed(rng)
