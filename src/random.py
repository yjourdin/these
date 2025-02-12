from abc import ABC
from typing import Any

from numpy.random import Generator, SeedSequence, default_rng

from .utils import tolist

Seed = int


def rng(seed: Seed | SeedSequence | None = None):
    return default_rng(seed)


def seed(rng: Generator = rng(), max: int = 2**63) -> Seed:
    return Seed(rng.integers(max))


def seeds(rng: Generator, nb: int = 1, max: int = 2**63) -> list[Seed]:
    return tolist(rng.integers(max, size=nb))


class Random(ABC):
    @classmethod
    def random(cls, rng: Generator, *args: Any, **kwargs: Any) -> Any: ...


class SeedMixin:
    def seed(self, *args: Any, **kwargs: Any) -> Seed:
        return getattr(self, "seed")

    def rng(self, *args: Any, **kwargs: Any):
        return rng(self.seed(*args, **kwargs))
