import csv
from math import exp
from time import time

from mcda.internal.core.interfaces import Learner
from numpy.random import Generator

from ..constants import DEFAULT_MAX_TIME
from .cooling_schedule import CoolingSchedule
from .neighbor import Neighbor
from .objective import Objective
from .type import Solution


class SimulatedAnnealing(Learner[Solution]):
    def __init__(
        self,
        T0: float,
        L: int,
        neighbor: Neighbor[Solution],
        objective: Objective[Solution],
        cooling_schedule: CoolingSchedule,
        init_sol: Solution,
        rng: Generator,
        Tf: float | None = None,
        max_time: int = DEFAULT_MAX_TIME,
        max_it: int | None = None,
        max_it_non_improving: int | None = None,
        log_file=None,
    ):
        self.T0 = T0
        self.L = L
        self.neighbor = neighbor
        self.objective = objective
        self.cooling_schedule = cooling_schedule
        self.Tf = Tf
        self.max_time = max_time
        self.max_it = max_it
        self.max_it_non_improving = max_it_non_improving
        self.init_sol = init_sol
        self.rng = rng
        self.log_file = log_file

    def _learn(
        self,
        initial_sol: Solution,
        rng: Generator,
    ):
        # Initialise
        temp = self.T0
        current_sol = initial_sol
        current_obj = self.objective(current_sol)
        self.best_sol = initial_sol
        self.best_obj = self.objective(self.best_sol)
        start_time = time()
        self.time = time() - start_time
        self.it = 0
        self.non_improving_it = 0

        if self.log_file:
            log_writer = csv.DictWriter(
                self.log_file,
                (
                    "It",
                    "Non-improving it",
                    "Time",
                    "Temp",
                    "Neighbor sol",
                    "Neighbor obj",
                    "Current obj",
                    "Best obj",
                ),
                dialect="unix",
            )
            log_writer.writeheader()

        # Stop when optimum reached
        if self.best_obj <= self.objective.optimum:
            return self.best_sol

        # Stopping criterion
        while (
            (not self.Tf or (temp > self.Tf))
            and (not self.max_time or (self.time < self.max_time))
            and (not self.max_it or (self.it < self.max_it))
            and (
                not self.max_it_non_improving
                or (self.non_improving_it < self.max_it_non_improving)
            )
        ):
            for _ in range(self.L):
                # New iteration
                self.time = time() - start_time
                self.it += 1
                self.non_improving_it += 1

                # Neighbor model
                neighbor_sol = self.neighbor(current_sol, rng)
                neighbor_obj = self.objective(neighbor_sol)

                prob: float
                if neighbor_obj <= current_obj:
                    prob = 1
                else:
                    try:
                        prob = exp((current_obj - neighbor_obj) / temp)
                    except OverflowError:
                        prob = 0

                if prob >= 1 or rng.random() < prob:
                    # Accepted
                    current_sol = neighbor_sol
                    current_obj = neighbor_obj

                    # New best
                    if current_obj < self.best_obj:
                        self.non_improving_it = 0
                        self.best_sol = current_sol
                        self.best_obj = current_obj

                        # Stop when optimum reached
                        if self.best_obj <= self.objective.optimum:
                            return self.best_sol

                if log_writer:
                    log_writer.writerow(
                        {
                            "It": self.it,
                            "Non-improving it": self.non_improving_it,
                            "Time": self.time,
                            "Temp": temp,
                            "Neighbor sol": neighbor_sol,
                            "Neighbor obj": neighbor_obj,
                            "Current obj": current_obj,
                            "Best obj": self.best_obj,
                        }
                    )

            # Update temperature
            temp = self.cooling_schedule(temp)
        return self.best_sol

    def learn(self):
        return self._learn(self.init_sol, self.rng)
