from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from numpy.random import Generator

from ..core.model import Model

T = TypeVar("T", bound=Model)


class Neighbor(Generic[T], ABC):
    @abstractmethod
    def __call__(self, model: T, rng: Generator) -> T:
        pass


class RandomNeighbor(Neighbor[T]):
    def __init__(self, neighbors: Sequence[Neighbor[T]]):
        self.neighbors = neighbors

    def __call__(self, model: T, rng: Generator) -> T:
        i = rng.integers(0, len(self.neighbors))
        return self.neighbors[i](model, rng)
