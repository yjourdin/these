from abc import abstractmethod
from pathlib import Path
from typing import Any, TypedDict

from mcda.internal.core.interfaces import Learner
from pulp import (  # pyright: ignore[reportMissingTypeStubs]
    LpProblem,
    LpSolver,
    LpVariable,
    getSolver,
    listSolvers,
)
from pulp import value as value_pulp  # pyright: ignore[reportMissingTypeStubs]

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, InitVar, dataclass, field
from src.random import SeedLike, int_


def value(x: Any):
    if (v := value_pulp(x)) is None:
        raise ValueError("None value")
    return v


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
    log_path: InitVar[Path | None] = None
    nb_cpus: InitVar[int] = 1

    def __post_init__(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        time_limit: float,
        seed: SeedLike | None,
        verbose: bool,
        log_path: Path,
        nb_cpus: int,
    ):
        self.create_solver(time_limit, int_(seed), verbose, nb_cpus, log_path)
        self.create_parameters()
        self.create_variables()
        self.create_problem()

    def learn(self):
        self.prob.solve(self.solver)
        try:
            self.create_solution()
        except ValueError:
            return None
        else:
            return self.sol

    def create_solver(
        self, time_limit: float, seed: int, verbose: bool, nb_cpus: int, log_path: Path
    ):
        kwargs: dict[str, Any] = {
            "msg": verbose,
            "threads": nb_cpus,
            "timeLimit": time_limit,
        }
        seed = seed % 2_000_000_000

        if "GUROBI" in listSolvers(True):
            kwargs["solver"] = "GUROBI"
            kwargs["seed"] = seed
            kwargs["logPath"] = str(log_path)
        elif "HiGHS" in listSolvers(True):
            kwargs["solver"] = "HiGHS"
            kwargs["random_seed"] = seed
            kwargs["log_file"] = str(log_path)
        else:
            kwargs["solver"] = "PULP_CBC_CMD"
            kwargs["options"] = [f"RandomS {seed}"]
            kwargs["logPath"] = str(log_path)

        self.solver = getSolver(**kwargs)

    @abstractmethod
    def create_parameters(self): ...

    @abstractmethod
    def create_variables(self): ...

    @abstractmethod
    def create_problem(self): ...

    @abstractmethod
    def create_solution(self): ...
