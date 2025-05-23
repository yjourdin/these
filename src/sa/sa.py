import csv
from dataclasses import InitVar, dataclass
from math import exp
from time import process_time
from typing import TextIO

from mcda.internal.core.interfaces import Learner

from ..constants import DEFAULT_MAX_TIME
from ..dataclass import Dataclass
from ..random import RNG, RNGParam, rng_
from .cooling_schedule import CoolingSchedule
from .neighbor import Neighbor
from .objective import Objective


@dataclass
class SimulatedAnnealing[S](Learner[S], Dataclass):
    T0: float
    L: int
    neighbor: Neighbor[S]
    objective: Objective[S]
    cooling_schedule: CoolingSchedule
    init_sol: S
    rng: InitVar[RNGParam] = None
    Tf: float | None = None
    max_time: int = DEFAULT_MAX_TIME
    max_it: int | None = None
    max_it_non_improving: int | None = None
    log_file: TextIO | None = None

    def __post_init__(self, rng: RNGParam):
        self._rng = rng_(rng)

    def _learn(
        self,
        initial_sol: S,
        rng: RNG,
    ):
        # Initialise
        temp = self.T0
        current_sol = initial_sol
        current_obj = self.objective(current_sol)
        self.best_sol = initial_sol
        self.best_obj = self.objective(self.best_sol)
        start_time = process_time()
        self.time = process_time() - start_time
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
            (self.time < self.max_time)
            and ((self.Tf is None) or (temp > self.Tf))
            and ((self.max_it is None) or (self.it < self.max_it))
            and (
                (self.max_it_non_improving is None)
                or (self.non_improving_it < self.max_it_non_improving)
            )
        ):
            for _ in range(self.L):
                # New iteration
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
                    except (OverflowError, ZeroDivisionError):
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

                # Update time
                self.time = process_time() - start_time

                if self.log_file:
                    log_writer.writerow({  # type: ignore
                        "It": self.it,
                        "Non-improving it": self.non_improving_it,
                        "Time": self.time,
                        "Temp": temp,
                        "Neighbor sol": neighbor_sol,
                        "Neighbor obj": neighbor_obj,
                        "Current obj": current_obj,
                        "Best obj": self.best_obj,
                    })

            # Update temperature
            temp = self.cooling_schedule(temp)
        return self.best_sol

    def learn(self):
        return self._learn(self.init_sol, self._rng)
