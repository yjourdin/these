from abc import ABC, abstractmethod
from dataclasses import dataclass

from mcda.relations import PreferenceStructure

from ..dataclass import Dataclass
from ..model import Model
from ..performance_table.type import PerformanceTableType


class Objective[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> float: ...

    @property
    @abstractmethod
    def optimum(self) -> float: ...


@dataclass
class FitnessObjective(Objective[Model], Dataclass):
    train_data: PerformanceTableType
    target: PreferenceStructure

    def __call__(self, sol: Model):
        return 1 - sol.fitness(self.train_data, self.target)

    @property
    def optimum(self) -> float:
        return 0
