from mcda.matrices import PerformanceTable
from mcda.relations import PreferenceStructure

from ..abstract_model import Model
from .sa import Objective


class FitnessObjective(Objective[Model]):
    def __init__(
        self, train_data: PerformanceTable, target: PreferenceStructure
    ) -> None:
        self.train_data = train_data
        self.target = target

    def __call__(self, model: Model) -> float:
        return 1 - model.fitness(self.train_data, self.target)

    @property
    def optimum(self) -> float:
        return 0
