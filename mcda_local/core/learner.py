from abc import abstractmethod
from typing import Generic, TypeVar

from .performance_table import PerformanceTable
from .relations import PreferenceStructure

T = TypeVar("T", covariant=True)


class Learner(Generic[T]):
    """This interface describes a generic learner."""

    @abstractmethod
    def learn(
        self, train_data: PerformanceTable, target: PreferenceStructure, **kwargs
    ) -> T | None:  # pragma: nocover
        """Learn and return an object.

        :return:
        """
        pass
