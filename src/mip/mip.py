from abc import abstractmethod
from typing import Any

from mcda.internal.core.interfaces import Learner
from pulp import LpProblem, getSolver, listSolvers

from ..constants import DEFAULT_MAX_TIME
from ..random import rng
from ..random import seed as random_seed


class MIP[T](Learner[T | None]):
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
        seed = seed if seed is not None else random_seed(rng())

        kwargs: dict[str, Any] = {"msg": verbose, "threads": 1, "timeLimit": time_limit}

        if "GUROBI" in listSolvers(True):
            kwargs["solver"] = "GUROBI"
            if seed is not None:
                kwargs["seed"] = seed % 2_000_000_000
        if "HiGHS" in listSolvers(True):
            kwargs["solver"] = "HiGHS"
            if seed is not None:
                kwargs["random_seed"] = seed % 2_000_000_000
        else:
            kwargs["solver"] = "PULP_CBC_CMD"
            if seed is not None:
                kwargs["options"] = [f"RandomS {seed % 2_000_000_000}"]

        self.solver = getSolver(**kwargs)

    def learn(self):
        self.create_problem()
        self.prob.solve(self.solver)
        return self.create_solution() if self.prob.sol_status > 0 else None

    @abstractmethod
    def create_problem(self, *args, **kwargs): ...

    @abstractmethod
    def create_solution(self) -> T: ...
