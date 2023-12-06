from abc import ABC, abstractmethod
from typing import Sequence

from .performance_table import PerformanceTable


class Selector(ABC):
    """Interface to implement selection MCDA algorithms."""

    @abstractmethod
    def select(
        self, performance_table: PerformanceTable, **kwargs
    ) -> Sequence:  # pragma: nocover
        """Select a subset of alternatives.

        :param performance_table:
        :return: selected alternatives
        """
        pass
