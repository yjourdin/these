from abc import ABC, abstractmethod
from typing import Generic

from mcda import PerformanceTable
from mcda.relations import PreferenceStructure

from ..model import Model
from .type import Solution


class Objective(Generic[Solution], ABC):
    @abstractmethod
    def __call__(self, sol: Solution) -> float: ...

    @property
    @abstractmethod
    def optimum(self) -> float: ...


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
