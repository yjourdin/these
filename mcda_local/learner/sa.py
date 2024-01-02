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
        T0: float,
        alpha: float,
        L: int,
        neighbor: Neighbor[T],
        Tf: float | None = None,
        max_time: int | None = None,
        max_iter: int | None = None,
        max_iter_non_improving: int | None = None,
    ):
        self.T0 = T0
        self.alpha = alpha
        self.L = L
        self.Tf = Tf
        self.max_time = max_time
        self.max_iter = max_iter
        self.max_iter_non_improving = max_iter_non_improving
        self.neighbor = neighbor

    def learn(
        self,
        train_data: PerformanceTable,
        target: PreferenceStructure,
        initial_model: T,
        rng: Generator,
    ):
        # Initialise
        temp = self.T0
        best_model = initial_model
        best_fitness = best_model.fitness(train_data, target)
        start_time = time()
        it = 0
        non_improving_it = 0

        # Stopping criterion
        while (
            (not self.Tf or (temp > self.Tf))
            and (not self.max_time or (time() < start_time + self.max_time))
            and (not self.max_iter or (it < self.max_iter))
            and (
                not self.max_iter_non_improving
                or (non_improving_it < self.max_iter_non_improving)
            )
        ):
            for _ in range(self.L):
                # New iteration
                it += 1
                non_improving_it += 1

                # Neighbor model
                neighbor_model = self.neighbor(best_model, rng)
                neighbor_fitness = neighbor_model.fitness(train_data, target)
                print(f"{neighbor_model}   {neighbor_fitness}   {temp}")

                if rng.random() < exp((neighbor_fitness - best_fitness) / temp):
                    # Accepted
                    if neighbor_fitness - best_fitness > 0:
                        non_improving_it = 0
                    best_model = neighbor_model
                    # print(f"{best_model}   {best_fitness}   {temp}")
                    best_fitness = neighbor_fitness

                # print(f"{best_fitness},{temp}")

                if best_fitness == 1:
                    return best_model

                # if it % 100 == 0:
                #     print(
                #         f"Iteration : {it} \t"
                #         f"Temperature : {temp:5.5f} \t"
                #         f"Best fitness : {best_fitness:3.3f}"
                #     )

            # Update temperature
            temp *= self.alpha
        return best_model
