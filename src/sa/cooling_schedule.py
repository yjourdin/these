from abc import ABC, abstractmethod

from src.dataclass import Dataclass, dataclass


class CoolingSchedule(ABC):
    @abstractmethod
    def __call__(self, temp: float) -> float: ...


@dataclass
class GeometricSchedule(CoolingSchedule, Dataclass):
    alpha: float

    def __call__(self, temp: float):
        return temp * self.alpha
