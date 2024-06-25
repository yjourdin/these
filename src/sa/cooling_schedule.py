from .sa import CoolingSchedule


class GeometricSchedule(CoolingSchedule):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, temp):
        return temp * self.alpha
