from abc import abstractmethod

from .model import Model
from .performance_table import PerformanceTable


class Assignator(Model):
    """Interface to implement assignment MCDA algorithms."""

    @abstractmethod
    def assign(
        self, performance_table: PerformanceTable, **kwargs
    ) -> dict:  # pragma: nocover
        """Assign alternatives to categories.

        :param performance_table:
        :return: categories
        """
        pass
