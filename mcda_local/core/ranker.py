from abc import abstractmethod
from typing import cast

from .model import Model
from .performance_table import PerformanceTable
from .relations import IndifferenceRelation, PreferenceRelation, PreferenceStructure
from .values import Ranking


class Ranker(Model):
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
        rank = cast(Ranking, self.rank(performance_table))
        s: int = 0
        for r in comparisons:
            a, b = r.elements
            match r:
                case PreferenceRelation():
                    s += (cast(int, rank.data[a]) > cast(int, rank.data[b]))
                case IndifferenceRelation():
                    s += (cast(int, rank.data[a]) == cast(int, rank.data[b]))
        return s / len(comparisons)
