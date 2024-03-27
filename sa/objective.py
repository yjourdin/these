from mcda.core.matrices import PerformanceTable
from mcda.core.relations import PreferenceStructure

from .sa import Objective


class FitnessObjective(Objective):
    def __init__(
        self, train_data: PerformanceTable, target: PreferenceStructure
    ) -> None:
        self.train_data = train_data
        self.target = target

    def __call__(self, model) -> float:
        return 1 - model.fitness(self.train_data, self.target)

    @property
    def optimum(self):
        return 0
