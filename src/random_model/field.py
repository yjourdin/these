from dataclasses import dataclass, field
from typing import Any

from ..field import RandomField, random_field
from ..random import RNGParam, seed


@random_field("seed")
@dataclass
class SeedField(RandomField):
    seed: int = field(default=seed())

    @staticmethod
    def field_random(
        rng: RNGParam = None,
        *args: Any,
        **kwargs: Any,
    ):
        return seed(rng)
