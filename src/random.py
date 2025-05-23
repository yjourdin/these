from abc import ABC
from collections.abc import Sequence
from typing import Any

import numpy as np

from .utils import tolist

type SeedLike = int | np.integer[Any] | Sequence[int] | np.random.SeedSequence
type RNG = np.random.Generator
type RNGLike = RNG | np.random.BitGenerator
type RNGParam = RNGLike | SeedLike | None


def rng_(rng: RNGParam = None) -> RNG:
    return np.random.default_rng(rng)


def seed(rng: RNGParam = None, max: int = 2**63):
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    else:
        return int(rng_(rng).integers(max))


def seeds(rng: RNGParam = None, nb: int = 1, max: int = 2**63):
    return tolist(rng_(rng).integers(max, size=nb))


class Random(ABC):
    @classmethod
    def random(cls, rng: RNG = rng_(), *args: Any, **kwargs: Any) -> Any: ...


class SeedMixin:
    def seed(self, *args: Any, **kwargs: Any) -> SeedLike:
        return getattr(self, "seed")

    def rng(self, *args: Any, **kwargs: Any):
        return rng_(self.seed(*args, **kwargs))
