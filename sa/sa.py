from abc import ABC, abstractmethod
from math import exp
from time import time
from typing import Generic, TypeVar

from mcda.internal.core.interfaces import Learner
from numpy.random import Generator

T = TypeVar("T")


class Neighbor(Generic[T], ABC):
    @abstractmethod
    def __call__(self, model: T, rng: Generator) -> T:
        pass


class Objective(Generic[T], ABC):
    @abstractmethod
    def __call__(self, sol: T) -> float:
        pass

    @property
    @abstractmethod
    def optimum(self) -> float:
        pass


class CoolingSchedule(ABC):
    @abstractmethod
    def __call__(self, temp: float) -> float:
        pass


class SimulatedAnnealing(Learner[T]):
    def __init__(
        self,
        T0: float,
        L: int,
        neighbor: Neighbor[T],
        objective: Objective,
        cooling_schedule: CoolingSchedule,
        initial_sol: T,
        rng: Generator,
        Tf: float | None = None,
        max_time: int | None = None,
        max_iter: int | None = None,
        max_iter_non_improving: int | None = None,
        verbose: bool = False,
    ):
        self.T0 = T0
        self.L = L
        self.neighbor = neighbor
        self.objective = objective
        self.cooling_schedule = cooling_schedule
        self.Tf = Tf
        self.max_time = max_time
        self.max_iter = max_iter
        self.max_iter_non_improving = max_iter_non_improving
        self.initial_sol = initial_sol
        self.rng = rng
        self.verbose = verbose

    def _learn(
        self,
        initial_sol: T,
        rng: Generator,
    ):
        # Initialise
        temp = self.T0
        current_sol = initial_sol
        current_objective = self.objective(current_sol)
        self.best_sol = initial_sol
        self.best_objective = self.objective(self.best_sol)
        start_time = time()
        self.time = time() - start_time
        self.it = 0
        self.non_improving_it = 0

        # Stopping criterion
        while (
            (not self.Tf or (temp > self.Tf))
            and (not self.max_time or (self.time < self.max_time))
            and (not self.max_iter or (self.it < self.max_iter))
            and (
                not self.max_iter_non_improving
                or (self.non_improving_it < self.max_iter_non_improving)
            )
        ):
            for _ in range(self.L):
                # New iteration
                self.time = time() - start_time
                self.it += 1
                self.non_improving_it += 1

                # Neighbor model
                neighbor_sol = self.neighbor(current_sol, rng)
                neighbor_objective = self.objective(neighbor_sol)

                prob: float
                if neighbor_objective <= current_objective:
                    prob = 1
                else:
                    try:
                        prob = exp((current_objective - neighbor_objective) / temp)
                    except OverflowError:
                        prob = 0

                if rng.random() < prob:
                    # Accepted
                    current_sol = neighbor_sol
                    current_objective = neighbor_objective

                    # New best
                    if current_objective < self.best_objective:
                        self.non_improving_it = 0
                        self.best_sol = current_sol
                        self.best_objective = current_objective

                        # Stop when fitness equals 1
                        if self.best_objective <= self.objective.optimum:
                            return self.best_sol

                if self.verbose:
                    print(
                        f"{self.it}   "
                        f"{self.non_improving_it}   "
                        f"{self.time}   "
                        f"{neighbor_sol}   "
                        f"{neighbor_objective:.3f}   "
                        f"{current_objective:.3f}   "
                        f"{self.best_objective:.3f}   "
                        f"{temp}"
                    )

            # Update temperature
            temp = self.cooling_schedule(temp)
        return self.best_sol

    def learn(self):
        return self._learn(self.initial_sol, self.rng)
