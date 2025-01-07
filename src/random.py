from abc import ABC, abstractmethod

from numpy.random import Generator, SeedSequence, default_rng

Seed = int


def rng(seed: Seed | SeedSequence | None = None):
    return default_rng(seed)


def seed(rng: Generator = rng(), max: int = 2**63) -> Seed:
    return Seed(rng.integers(max))


def seeds(rng: Generator, nb: int = 1, max: int = 2**63) -> list[Seed]:
    return rng.integers(max, size=nb).tolist()


class Random(ABC):
    @classmethod
    def random(cls, rng: Generator, *args, **kwargs): ...


class SeedMixin:
    @abstractmethod
    def seed(self, *args, **kwargs) -> Seed: ...

    def rng(self, *args, **kwargs):
        return rng(self.seed(*args, **kwargs))
