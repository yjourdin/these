from abc import ABC, abstractmethod

from .performance_table import PerformanceTable


class Clusterizor(ABC):
    """Interface to implement clustering MCDA algorithms."""

    @abstractmethod
    def clusterize(
        self, performance_table: PerformanceTable, **kwargs
    ) -> dict:  # pragma: nocover
        """Clusterize alternatives.

        :param performance_table:
        :return: alternatives clusters
        """
        pass
