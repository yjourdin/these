from abc import abstractmethod
from typing import Any, TypeVar

from mcda.internal.core.interfaces import Learner
from numpy.random import default_rng
from pulp import LpProblem, getSolver, listSolvers

from ..constants import DEFAULT_MAX_TIME
from ..seed import seed as random_seed

T = TypeVar("T")


class MIP(Learner[T | None]):
    def __init__(
        self,
        time_limit: int = DEFAULT_MAX_TIME,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.var: dict[str, Any] = {}
        self.param: dict[str, Any] = {}
        self.prob = LpProblem()
        self.objective = None
        seed = seed if seed is not None else random_seed(default_rng())

        kwargs: dict[str, Any] = {"msg": verbose, "threads": 1, "timeLimit": time_limit}

        if "GUROBI" in listSolvers(True):
            kwargs["solver"] = "GUROBI"
            if seed is not None:
                kwargs["seed"] = seed % 2_000_000_000
            self.solver = getSolver(**kwargs)
        else:
            kwargs["solver"] = "PULP_CBC_CMD"
            kwargs["options"] = [f"RandomS {seed % 2_000_000_000}"]
            self.solver = getSolver(**kwargs)

    def learn(self):
        self.create_problem()
        self.prob.solve(self.solver)
        return self.create_solution()

    @abstractmethod
    def create_problem(self, *args, **kwargs): ...

    @abstractmethod
    def create_solution(self): ...
