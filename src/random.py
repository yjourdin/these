from abc import ABC
from collections.abc import Sequence
from typing import Any

import numpy as np

type Seed = np.random.SeedSequence
type SeedLike = Seed | int | np.integer[Any] | Sequence[int]
type RNG = np.random.Generator
type RNGParam = RNG | SeedLike | None


def rng_(rng: RNGParam = None) -> RNG:
    return np.random.default_rng(rng)


def seed_(rng: RNGParam = None, max: int = 2**63) -> Seed:
    return (
        rng
        if isinstance(rng, np.random.SeedSequence)
        else np.random.SeedSequence(int_(rng, max))
    )


def int_(rng: RNGParam = None, max: int = 2**63) -> int:
    match rng:
        case int() | np.integer():
            return int(rng)
        case np.random.Generator():
            return int(rng.integers(max))
        case np.random.SeedSequence():
            return int_(rng.entropy)
        case _:
            return int_(rng_(rng))


class Random(ABC):
    @classmethod
    def random(cls, rng: RNGParam = None, *args: Any, **kwargs: Any) -> Any: ...


class SeedMixin:
    def seed(self, *args: Any, **kwargs: Any):
        return seed_(getattr(self, "seed"))  # noqa: B009

    def rng(self, *args: Any, **kwargs: Any):
        return rng_(self.seed(*args, **kwargs))
