from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

from mcda.internal.core.scales import Scale
from mcda.internal.core.values import Ranking
from mcda.matrices import PerformanceTable
from mcda.relations import PreferenceStructure

from .preference_structure.fitness import fitness_comparisons_ranking

S = TypeVar("S", bound=Scale, covariant=True)


@dataclass
class Model(ABC, Generic[S]):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    def from_dict(cls, dct: dict) -> "Model":
        return cls(**dct)

    @classmethod
    @abstractmethod
    def from_json(cls, s: str) -> "Model":
        pass

    def to_dict(self):
        return asdict(self)

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
        return fitness_comparisons_ranking(comparisons, ranking)
