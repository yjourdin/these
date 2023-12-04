from math import exp
from typing import TypeVar

from numpy.random import Generator

from ..core.learner import Learner
from ..core.performance_table import PerformanceTable
from ..core.ranker import Ranker
from ..core.relations import PreferenceStructure
from .neighbor import Neighbor

T = TypeVar("T", bound=Ranker)


class SimulatedAnnealing(Learner[T]):
    def __init__(
        self,
        temp_initial: float,
        temp_max: float,
        alpha: float,
        L: int,
        neighbor: Neighbor[T],
    ):
        self.temp_initial = temp_initial
        self.temp_max = temp_max
        self.alpha = alpha
        self.L = L
        self.neighbor = neighbor

    def learn(
        self,
        train_data: PerformanceTable,
        target: PreferenceStructure,
        initial_model: T,
        rng: Generator,
    ):
        temp = self.temp_initial
        best_model = initial_model
        best_fitness = best_model.fitness(train_data, target)
        while temp > self.temp_max:
            for _ in range(self.L):
                neighbor_model = self.neighbor(best_model, rng)
                neighbor_fitness = neighbor_model.fitness(train_data, target)
                if rng.random() < exp((best_fitness - neighbor_fitness) / temp):
                    best_model = neighbor_model
                    best_fitness = neighbor_fitness
            temp = self.alpha * temp
            print(f"Temperature : {temp}")
        return best_model
