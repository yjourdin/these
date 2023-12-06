from abc import ABC, abstractmethod
from typing import cast

from .performance_table import PerformanceTable
from .relations import IndifferenceRelation, PreferenceRelation, PreferenceStructure
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
        self, performance_table: PerformanceTable, comparisons: PreferenceStructure
    ) -> float:
        ranking = cast(Ranking, self.rank(performance_table))
        s: int = 0
        for r in comparisons:
            a, b = r.elements
            match r:
                case PreferenceRelation():
                    s += cast(int, ranking.data[a]) > cast(int, ranking.data[b])
                case IndifferenceRelation():
                    s += cast(int, ranking.data[a]) == cast(int, ranking.data[b])
        return s / len(comparisons)
