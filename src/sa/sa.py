import csv
from collections.abc import Callable
from dataclasses import InitVar
from math import exp
from time import process_time
from typing import ClassVar, TextIO

from mcda.internal.core.interfaces import Learner

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, dataclass
from src.random import RNG, RNGParam, rng_
from src.utils import none_guard

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
    log_file: InitVar[TextIO | None] = None

    stopping_criteria_dict: ClassVar[dict[str, str]] = {
        "max_time": "stop_time",
        "Tf": "stop_temp",
        "max_it": "stop_it",
        "max_it_non_improving": "stop_it_non_improving",
    }

    def __post_init__(self, rng: RNGParam, log_file: TextIO | None):
        self.stopping_criteria: list[Callable[[], bool]] = [self.stop_optimum]
        for attr, f in self.stopping_criteria_dict.items():
            if getattr(self, attr) is not None:
                self.stopping_criteria.append(getattr(self, f))
        self._rng = rng_(rng)
        if log_file:
            self.log_writer = csv.DictWriter(
                log_file,
                (
                    "It",
                    "Non-improving it",
                    "Time",
                    "Temp",
                    "Neighbor sol",
                    "Current sol",
                    "Best sol",
                    "Neighbor obj",
                    "Current obj",
                    "Best obj",
                ),
                dialect="unix",
            )
            self.log_writer.writeheader()

    def stop_time(self):
        return self.time >= self.max_time

    def stop_optimum(self):
        return self.best_obj <= self.objective.optimum

    def stop_temp(self):
        if none_guard(self.Tf):
            return self.temp <= self.Tf
        else:
            return False

    def stop_it(self):
        if none_guard(self.max_it):
            return self.it >= self.max_it
        else:
            return False

    def stop_it_non_improving(self):
        if none_guard(self.max_it_non_improving):
            return self.non_improving_it >= self.max_it_non_improving
        else:
            return False


    def init(self, initial_sol: S):
        self.temp = self.T0
        self.current_sol = initial_sol
        self.current_obj = self.objective(self.current_sol)
        self.best_sol = initial_sol
        self.best_obj = self.objective(self.best_sol)
        self.start_time = process_time()
        self.time = process_time() - self.start_time
        self.it = 0
        self.non_improving_it = 0

    def stop(self):
        return any(f() for f in self.stopping_criteria)

    def main_loop(self, rng: RNG):
        while not self.stop():
            for _ in range(self.L):
                # New iteration
                self.it += 1
                self.non_improving_it += 1

                # Neighbor model
                neighbor_sol = self.neighbor(self.current_sol, rng)
                neighbor_obj = self.objective(neighbor_sol)

                if hasattr(self, "log_writer"):
                    self.log_writer.writerow({
                        "It": self.it,
                        "Non-improving it": self.non_improving_it,
                        "Time": self.time,
                        "Temp": self.temp,
                        "Neighbor sol": neighbor_sol,
                        "Current sol": self.current_sol,
                        "Best sol": self.best_sol,
                        "Neighbor obj": neighbor_obj,
                        "Current obj": self.current_obj,
                        "Best obj": self.best_obj,
                    })

                prob: float
                if neighbor_obj <= self.current_obj:
                    prob = 1
                else:
                    try:
                        prob = exp((self.current_obj - neighbor_obj) / self.temp)
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
                self.time = process_time() - self.start_time

            # Update temperature
            self.temp = self.cooling_schedule(self.temp)
        return self.best_sol


    def _learn(
        self,
        initial_sol: S,
        rng: RNG,
    ):
        # Init
        self.init(initial_sol)

        # Main loop
        return self.main_loop(rng)


    def learn(self):
        return self._learn(self.init_sol, self._rng)
