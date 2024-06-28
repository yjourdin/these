import csv
from time import time

from mcda.internal.core.interfaces import Learner
from numpy.random import Generator

from .neighbor import Neighbor
from .objective import Objective
from .type import T


class RandomWalk(Learner[T]):
    def __init__(
        self,
        neighbor: Neighbor[T],
        objective: Objective,
        init_sol: T,
        rng: Generator,
        max_time: int | None = None,
        max_it: int | None = None,
        log_file=None,
    ):
        self.neighbor = neighbor
        self.objective = objective
        self.max_time = max_time
        self.max_it = max_it
        self.initial_sol = init_sol
        self.rng = rng
        self.log_file = log_file

    def _learn(
        self,
        initial_sol: T,
        rng: Generator,
    ):
        # Initialise
        sol = initial_sol
        obj = self.objective(sol)
        start_time = time()
        self.time = time() - start_time
        self.it = 0
        self.non_improving_it = 0
        log_writer = (
            csv.DictWriter(
                self.log_file,
                (
                    "It",
                    "Non-improving it",
                    "Time",
                    "Sol",
                    "Obj",
                ),
                dialect="unix",
            )
            if self.log_file
            else None
        )
        if log_writer:
            log_writer.writeheader()

        # Stopping criterion
        while (not self.max_time or (self.time < self.max_time)) and (
            not self.max_it or (self.it < self.max_it)
        ):
            # New iteration
            self.time = time() - start_time
            self.it += 1
            self.non_improving_it += 1

            # Neighbor model
            sol = self.neighbor(sol, rng)
            obj = self.objective(sol)

            if log_writer:
                log_writer.writerow(
                    {
                        "It": self.it,
                        "Non-improving it": self.non_improving_it,
                        "Time": self.time,
                        "Sol": sol,
                        "Obj": obj,
                    }
                )
        return sol

    def learn(self):
        return self._learn(self.initial_sol, self.rng)
