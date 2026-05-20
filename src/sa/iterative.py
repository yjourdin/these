import csv
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import InitVar
from pathlib import Path
from time import thread_time
from typing import ClassVar, NotRequired, TypedDict

from mcda.internal.core.interfaces import Learner

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, dataclass
from src.random import RNG, RNGParam, rng_
from src.utils import file_or_stdout, none_guard

from .neighbor import Neighbor
from .objective import Objective


@dataclass
class Iterative[S](Learner[S], Dataclass):
    neighbor: Neighbor[S]
    objective: Objective[S]
    init_sol: S
    rng: InitVar[RNGParam] = None
    max_time: int = DEFAULT_MAX_TIME
    max_it: int | None = None
    max_it_non_improving: int | None = None
    verbose: bool = False
    log_path: Path | None = None
    stopping_criteria_dict: ClassVar[dict[str, str]] = {
        "max_time": "stop_time",
        "max_it": "stop_it",
        "max_it_non_improving": "stop_it_non_improving",
    }

    class LogFields[T](TypedDict):
        It: int
        Non_improving_it: int
        Time: float
        Neighbor_sol: NotRequired[T]
        Current_sol: T
        Best_sol: NotRequired[T]
        Neighbor_obj: NotRequired[float]
        Current_obj: float
        Best_obj: NotRequired[float]

    def __post_init__(self, rng: RNGParam):
        self.stopping_criteria: list[Callable[[], bool]] = [self.stop_optimum]
        for attr, f in self.stopping_criteria_dict.items():
            if getattr(self, attr) is not None:
                self.stopping_criteria.append(getattr(self, f))
        self._rng = rng_(rng)

    @contextmanager
    def log_writer(self):
        with file_or_stdout(self.log_path, "w", "") as f:
            yield csv.DictWriter(
                f,
                list(self.LogFields.__annotations__.keys()),
                dialect="unix",
            )

    def stop_time(self):
        return self.time >= self.max_time

    def stop_optimum(self):
        return self.best_obj <= self.objective.optimum

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
        self.current_sol = initial_sol
        self.current_obj = self.objective(self.current_sol)
        self.best_sol = initial_sol
        self.best_obj = self.objective(self.best_sol)
        self.start_time = thread_time()
        self.time = thread_time() - self.start_time
        self.it = 0
        self.non_improving_it = 0

        if self.verbose:
            with self.log_writer() as log_writer:
                log_writer.writeheader()

    def stop(self):
        return any(f() for f in self.stopping_criteria)

    def main_loop(self, rng: RNG) -> S: ...

    def learn(self):
        # Init
        self.init(self.init_sol)

        # Main loop
        return self.main_loop(self._rng)
