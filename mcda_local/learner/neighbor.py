from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from numpy import array
from numpy.random import Generator

T = TypeVar("T")


class Neighbor(Generic[T], ABC):
    @abstractmethod
    def __call__(self, model: T, rng: Generator) -> T:
        pass


class RandomNeighbor(Neighbor[T]):
    def __init__(
        self, neighbors: Sequence[Neighbor[T]], prob: Sequence[float] | None = None
    ):
        self.neighbors = neighbors
        if prob:
            prob_array = array(prob)
            self.prob = prob_array / prob_array.sum()
        else:
            self.prob = None

    def __call__(self, model: T, rng: Generator) -> T:
        i = rng.choice(len(self.neighbors), p=self.prob)
        return self.neighbors[i](model, rng)
