from abc import ABC, abstractmethod


class CoolingSchedule(ABC):
    @abstractmethod
    def __call__(self, temp: float) -> float:
        ...


class GeometricSchedule(CoolingSchedule):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, temp):
        return temp * self.alpha
