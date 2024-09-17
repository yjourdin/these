from abc import abstractmethod

from numpy.random import Generator, SeedSequence, default_rng

Seed = int


def rng(seed: Seed | SeedSequence | None = None):
    return default_rng(seed)


def seed(rng: Generator = rng(), max: int = 2**63) -> Seed:
    return rng.integers(max)


def seeds(rng: Generator, nb: int = 1, max: int = 2**63) -> list[Seed]:
    return rng.integers(max, size=nb).tolist()


class SeedMixin:
    @property
    @abstractmethod
    def seed(self) -> Seed: ...

    @property
    def rng(self):
        return rng(self.seed)
