"""This module is used to gather core interfaces and encourage their use for
a more coherent API.
"""

from abc import ABC, abstractmethod


class Model(ABC):
    """Interface to implement objects that can be copied."""

    @abstractmethod
    def copy(self) -> "Model":
        """Copy self

        :return:
        """
        pass
