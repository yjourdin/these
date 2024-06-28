from abc import ABC, abstractmethod
from typing import Generic

from mcda.matrices import PerformanceTable
from mcda.relations import PreferenceStructure

from ..abstract_model import Model
from .type import T


class Objective(Generic[T], ABC):
    @abstractmethod
    def __call__(self, sol: T) -> float:
        pass

    @property
    @abstractmethod
    def optimum(self) -> float:
        pass


class FitnessObjective(Objective[Model]):
    def __init__(
        self, train_data: PerformanceTable, target: PreferenceStructure
    ) -> None:
        self.train_data = train_data
        self.target = target

    def __call__(self, sol):
        return 1 - sol.fitness(self.train_data, self.target)

    @property
    def optimum(self) -> float:
        return 0
