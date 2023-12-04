from abc import abstractmethod

from .model import Model
from .performance_table import PerformanceTable


class Clusterizor(Model):
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
