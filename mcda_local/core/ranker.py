from abc import abstractmethod

from .model import Model
from .performance_table import PerformanceTable
from .relations import PreferenceStructure


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
        rank = self.rank(performance_table)
        return sum(
            rel in rank.preference_structure.transitive_closure for rel in comparisons
        ) / len(comparisons.relations)
