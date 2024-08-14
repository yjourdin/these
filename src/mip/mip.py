from typing import Any, TypeVar

from mcda.internal.core.interfaces import Learner
from numpy.random import default_rng
from pulp import LpProblem, getSolver, listSolvers

from ..seed import seed as random_seed

T = TypeVar("T")


class MIP(Learner[T | None]):
    def __init__(
        self,
        time_limit: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.prob = LpProblem()
        seed = seed if seed is not None else random_seed(default_rng())

        kwargs: dict[str, Any] = {"msg": verbose, "threads": 1}
        if time_limit is not None:
            kwargs["timeLimit"] = time_limit

        if "GUROBI" in listSolvers(True):
            kwargs["solver"] = "GUROBI"
            if seed is not None:
                kwargs["seed"] = seed % 2_000_000_000
            self.solver = getSolver(**kwargs)
        else:
            kwargs["solver"] = "PULP_CBC_CMD"
            self.solver = getSolver(**kwargs)
