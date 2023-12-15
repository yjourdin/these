from abc import ABC, abstractmethod
from typing import cast

from .performance_table import PerformanceTable
from .relations import (
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
    Relation,
)
from .values import Ranking


class Ranker(ABC):
    """Interface to implement ranking MCDA algorithms."""

    @abstractmethod
    def rank(
        self, performance_table: PerformanceTable, **kwargs
    ) -> PreferenceStructure:  # pragma: nocover
        """Rank alternatives.

        :param performance_table:
        :return: ranking
        """
        pass

    def fitness(
        self, performance_table: PerformanceTable, comparisons: list[Relation]
    ) -> float:
        ranking = cast(Ranking, self.rank(performance_table))
        ranking_dict = ranking.data.to_dict()
        s: int = 0
        for r in comparisons:
            a, b = r.elements
            match r:
                case PreferenceRelation():
                    s += cast(int, ranking_dict[a]) < cast(int, ranking_dict[b])
                case IndifferenceRelation():
                    s += cast(int, ranking_dict[a]) == cast(int, ranking_dict[b])
        return s / len(comparisons)
