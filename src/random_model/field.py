from dataclasses import dataclass, field

from numpy.random import Generator

from ..field import GeneratedField, generated_field
from ..random import Seed, seed


@generated_field("seed")
@dataclass
class SeedField(GeneratedField):
    seed: Seed = field(default=seed())

    @staticmethod
    def field_random(
        rng: Generator,
        *args,
        **kwargs,
    ):
        return seed(rng)

    @staticmethod
    def field_balanced(
        *args,
        **kwargs,
    ):
        return 0
