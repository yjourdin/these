from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mcda.core.matrices import PerformanceTable
from mcda.core.relations import (
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
)
from mcda.core.scales import Scale
from mcda.core.values import Ranking

S = TypeVar("S", bound=Scale, covariant=True)


def fitness(ranking: Ranking, comparisons: PreferenceStructure):
    # ranking_dict = ranking.data.to_dict()
    s: int = 0
    for r in comparisons:
        a, b = r.elements
        match r:
            case PreferenceRelation():
                # s += cast(int, ranking_dict[a]) < cast(int, ranking_dict[b])
                s += ranking[a] < ranking[b]
            case IndifferenceRelation():
                # s += cast(int, ranking_dict[a]) == cast(int, ranking_dict[b])
                s += ranking[a] == ranking[b]
    return s / len(comparisons)


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
        return fitness(ranking, comparisons)
