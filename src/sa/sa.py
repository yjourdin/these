from math import exp
from time import thread_time
from typing import ClassVar

from src.dataclass import dataclass
from src.random import RNG
from src.utils import none_guard

from .cooling_schedule import CoolingSchedule, GeometricSchedule
from .iterative import Iterative


@dataclass
class SimulatedAnnealing[S](Iterative[S]):
    T0: float = 1
    L: int = 1
    cooling_schedule: CoolingSchedule = GeometricSchedule(0.999)
    Tf: float | None = None
    stopping_criteria_dict: ClassVar[dict[str, str]] = (
        Iterative.stopping_criteria_dict | {"Tf": "stop_temp"}
    )

    class LogFields[T](Iterative.LogFields[T]):
        Temp: float

    def stop_temp(self):
        if none_guard(self.Tf):
            return self.temp <= self.Tf
        else:
            return False

    def init(self, initial_sol: S):
        super().init(initial_sol)
        self.temp = self.T0

    def main_loop(self, rng: RNG):
        while not self.stop():
            for _ in range(self.L):
                # New iteration
                self.time = thread_time() - self.start_time
                self.it += 1
                self.non_improving_it += 1

                # Neighbor model
                neighbor_sol = self.neighbor(self.current_sol, rng)
                neighbor_obj = self.objective(neighbor_sol)

                if hasattr(self, "log_writer"):
                    self.log_writer.writerow(
                        self.LogFields(
                            It=self.it,
                            Non_improving_it=self.non_improving_it,
                            Time=self.time,
                            Temp=self.temp,
                            Neighbor_sol=neighbor_sol,
                            Current_sol=self.current_sol,
                            Best_sol=self.best_sol,
                            Neighbor_obj=neighbor_obj,
                            Current_obj=self.current_obj,
                            Best_obj=self.best_obj,
                        )
                    )

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
                self.time = thread_time() - self.start_time

            # Update temperature
            self.temp = self.cooling_schedule(self.temp)
        return self.best_sol
