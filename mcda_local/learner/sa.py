from math import exp
from time import time
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
        alpha: float,
        L: int,
        neighbor: Neighbor[T],
        temp_final: float | None = None,
        time_limit: int | None = None,
        iteration_limit: int | None = None,
        non_improving_limit: int | None = None,
    ):
        self.temp_initial = temp_initial
        self.alpha = alpha
        self.L = L
        self.temp_final = temp_final
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.non_improving_limit = non_improving_limit
        self.neighbor = neighbor

    def learn(
        self,
        train_data: PerformanceTable,
        target: PreferenceStructure,
        initial_model: T,
        rng: Generator,
    ):
        # Initialise
        temp = self.temp_initial
        best_model = initial_model
        best_fitness = best_model.fitness(train_data, target)
        start_time = time()
        it = 0
        non_improving_it = 0

        # Stopping criterion
        while (
            (self.temp_final and (temp > self.temp_final))
            and (self.time_limit and (time() < start_time + self.time_limit))
            and (self.iteration_limit and (it < self.iteration_limit))
            and (
                self.non_improving_limit
                and (non_improving_it < self.non_improving_limit)
            )
        ):
            for _ in range(self.L):
                # New iteration
                it += 1
                non_improving_it += 1

                # Neighbor model
                neighbor_model = self.neighbor(best_model, rng)
                neighbor_fitness = neighbor_model.fitness(train_data, target)

                if rng.random() < exp((neighbor_fitness - best_fitness) / temp):
                    # Accepted
                    best_model = neighbor_model
                    best_fitness = neighbor_fitness
                    non_improving_it = 0

            # Update temperature
            temp *= self.alpha
        return best_model
