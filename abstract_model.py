from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mcda.internal.core.scales import Scale
from mcda.internal.core.values import Ranking
from mcda.matrices import PerformanceTable
from mcda.relations import PreferenceStructure

from fitness import fitness_comparisons

S = TypeVar("S", bound=Scale, covariant=True)


class Model(ABC, Generic[S]):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, s: str) -> "Model":
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    @abstractmethod
    def rank(self, performance_table: PerformanceTable[S]) -> Ranking:
        pass

    def fitness(
        self, performance_table: PerformanceTable[S], comparisons: PreferenceStructure
    ) -> float:
        ranking = self.rank(performance_table)
        return fitness_comparisons(ranking, comparisons)
