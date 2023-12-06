from abc import ABC, abstractmethod

from .performance_table import PerformanceTable


class Assignator(ABC):
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
