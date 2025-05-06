from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Any, TypedDict

from mcda.internal.core.interfaces import Learner
from pulp import LpProblem, LpSolver, LpVariable, getSolver, listSolvers  # type: ignore

from ..constants import DEFAULT_MAX_TIME
from ..dataclass import Dataclass
from ..random import rng
from ..random import seed as random_seed

type D[T: LpVariable | D] = dict[Any, T] # type: ignore


class MIPVars(TypedDict): ...


@dataclass
class MIPParams(Dataclass): ...


@dataclass
class MIP[T, Vars: MIPVars, Params: MIPParams](Learner[T | None], Dataclass):
    vars: Vars = field(init=False)
    params: Params = field(init=False)
    prob: LpProblem = field(init=False)
    objective: float | None = field(init=False)
    solver: LpSolver = field(init=False)
    time_limit: InitVar[float]
    seed: InitVar[int | None]
    verbose: InitVar[bool]

    def __post_init__(
        self,
        time_limit: float = DEFAULT_MAX_TIME,
        seed: int | None = None,
        verbose: bool = False,
        *args: Any,
        **kw: Any,
    ):
        self.prob = LpProblem()
        self.objective = None
        seed = seed if seed is not None else random_seed(rng())

        kwargs: dict[str, Any] = {"msg": verbose, "threads": 1, "timeLimit": time_limit}

        if "GUROBI" in listSolvers(True):
            kwargs["solver"] = "GUROBI"
            kwargs["seed"] = seed % 2_000_000_000
        elif "HiGHS" in listSolvers(True):
            kwargs["solver"] = "HiGHS"
            kwargs["random_seed"] = seed % 2_000_000_000
        else:
            kwargs["solver"] = "PULP_CBC_CMD"
            kwargs["options"] = [f"RandomS {seed % 2_000_000_000}"]

        self.solver = getSolver(**kwargs)

    def learn(self):
        self.create_problem()
        self.prob.solve(self.solver)
        return self.create_solution() if self.prob.sol_status > 0 else None

    @abstractmethod
    def create_problem(self, *args: Any, **kwargs: Any): ...

    @abstractmethod
    def create_solution(self) -> T: ...
