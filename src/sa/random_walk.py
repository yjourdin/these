import csv
import io
from dataclasses import dataclass
from time import process_time

from mcda.internal.core.interfaces import Learner
from numpy.random import Generator

from ..dataclass import Dataclass
from .neighbor import Neighbor
from .objective import Objective


@dataclass
class RandomWalk[S](Learner[S], Dataclass):
    neighbor: Neighbor[S]
    objective: Objective[S]
    init_sol: S
    rng: Generator
    max_time: int | None = None
    max_it: int | None = None
    log_file: io.StringIO | None = None

    def _learn(
        self,
        initial_sol: S,
        rng: Generator,
    ):
        # Initialise
        sol = initial_sol
        obj = self.objective(sol)
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
                    "Sol",
                    "Obj",
                ),
                dialect="unix",
            )
            log_writer.writeheader()

        # Stopping criterion
        while ((self.max_time is None) or (self.time < self.max_time)) and (
            (self.max_it is None) or (self.it < self.max_it)
        ):
            # New iteration
            self.time = process_time() - start_time
            self.it += 1
            self.non_improving_it += 1

            # Neighbor model
            sol = self.neighbor(sol, rng)
            obj = self.objective(sol)

            if self.log_file:
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
        return self._learn(self.init_sol, self.rng)
