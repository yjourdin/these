from abc import abstractmethod
from typing import Any, TypedDict

from mcda.internal.core.interfaces import Learner
from pulp import LpProblem, LpSolver, LpVariable, getSolver, listSolvers  # type: ignore

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, InitVar, dataclass, field
from src.random import SeedLike, int_

type D[T: LpVariable | D] = dict[Any, T]


class MIPVars(TypedDict): ...


@dataclass
class MIPParams(Dataclass): ...


@dataclass(kw_only=True)
class MIP[T, Vars: MIPVars, Params: MIPParams](Learner[T | None], Dataclass):
    vars: Vars = field(init=False)
    params: Params = field(init=False)
    prob: LpProblem = field(init=False)
    solver: LpSolver = field(init=False)
    sol: T = field(init=False)
    time_limit: InitVar[float] = DEFAULT_MAX_TIME
    seed: InitVar[SeedLike | None] = None
    verbose: InitVar[bool] = False
    nb_cpus: InitVar[int] = 1

    def __post_init__(
        self,
        time_limit: float,
        seed: SeedLike | None,
        verbose: bool,
        nb_cpus: int,
        *args: Any,
        **kw: Any,
    ):
        self.create_solver(time_limit, int_(seed), verbose, nb_cpus)

    def learn(self):
        self.create_parameters()
        self.create_variables()
        self.create_problem()
        self.prob.solve(self.solver)
        try:
            self.create_solution()
        except TypeError:
            return None
        else:
            return self.sol

    def create_solver(self, time_limit: float, seed: int, verbose: bool, nb_cpus: int):
        kwargs: dict[str, Any] = {
            "msg": verbose,
            "threads": nb_cpus,
            "timeLimit": time_limit,
        }

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

    @abstractmethod
    def create_parameters(self): ...

    @abstractmethod
    def create_variables(self): ...

    @abstractmethod
    def create_problem(self): ...

    @abstractmethod
    def create_solution(self): ...
