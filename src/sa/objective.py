from abc import ABC, abstractmethod
from dataclasses import dataclass

from mcda import PerformanceTable
from mcda.relations import PreferenceStructure

from ..dataclass import Dataclass
from ..model import Model


class Objective[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> float: ...

    @property
    @abstractmethod
    def optimum(self) -> float: ...


@dataclass
class FitnessObjective(Objective[Model], Dataclass):
    train_data: PerformanceTable
    target: PreferenceStructure

    def __call__(self, sol):
        return 1 - sol.fitness(self.train_data, self.target)

    @property
    def optimum(self) -> float:
        return 0
