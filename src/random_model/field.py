from dataclasses import dataclass, field
from typing import Any

from src.field import RandomField, random_field
from src.random import RNGParam, Seed, seed_


@random_field("seed")
@dataclass
class SeedField(RandomField):
    seed: Seed = field(default=seed_())

    @staticmethod
    def field_random(
        rng: RNGParam = None,
        *args: Any,
        **kwargs: Any,
    ):
        return seed_(rng)
