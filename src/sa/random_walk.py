from src.dataclass import dataclass
from src.random import RNG

from ..utils import catchtime
from .iterative import Iterative


@dataclass
class RandomWalk[S](Iterative[S]):
    def main_loop(self, rng: RNG):
        while not self.stop():
            # New iteration
            self.it += 1
            self.non_improving_it += 1

            with catchtime() as time:
                # Neighbor model
                self.current_sol = self.neighbor(self.current_sol, rng)
                self.current_obj = self.objective(self.current_sol)

            self.time += time()

            if self.verbose:
                with self.log_writer() as log_writer:
                    log_writer.writerow(
                        self.LogFields(
                            It=self.it,
                            Non_improving_it=self.non_improving_it,
                            Time=self.time,
                            Current_sol=self.current_sol,
                            Current_obj=self.current_obj,
                        )
                    )
        return self.current_sol
