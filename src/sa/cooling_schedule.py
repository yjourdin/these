from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..dataclass import Dataclass


class CoolingSchedule(ABC):
    @abstractmethod
    def __call__(self, temp: float) -> float: ...


@dataclass
class GeometricSchedule(CoolingSchedule, Dataclass):
    alpha: float

    def __call__(self, temp: float):
        return temp * self.alpha
