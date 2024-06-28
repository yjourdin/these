from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from multiprocessing import Queue
from typing import Any, ClassVar, Literal

from numpy.random import Generator, default_rng

from ..model import ModelType
from .config import Config, MIPConfig, SAConfig
from .job import (
    create_A_test,
    create_A_train,
    create_D,
    create_Mo,
    run_MIP,
    run_SA,
    run_test,
)
from .directory import Directory
from .seed import Seeds


def print_attribute(name: str, value: Any | None, width: int) -> str:
    if value is not None:
        if isinstance(value, Config):
            value = value.id
        return f"{name.capitalize()}: {value:{width}}"
    else:
        return f"{'':{len(name)}}  {'':{width}}"


WIDTH = {
    "m": 2,
    "n_tr": 4,
    "Atr_id": 2,
    "Mo": 4,
    "ko": 2,
    "Mo_id": 2,
    "n": 4,
    "error": 4,
    "D_id": 2,
    "Me": 4,
    "ke": 2,
    "method": 3,
    "config": 4,
    "Me_id": 2,
    "n_te": 4,
    "Ate_id": 2,
}


@dataclass(frozen=True)
class Task(ABC):
    name: ClassVar[str]
    seeds: InitVar[Seeds]

    def __str__(self) -> str:
        attributes = (
            print_attribute(attr, getattr(self, attr, None), width)
            for attr, width in WIDTH.items()
        )
        return f"{self.name:7} ({', '.join(attributes)})"

    @abstractmethod
    def __call__(
        self,
        dir: Directory,
        queues: dict[str, Queue],
    ) -> None:
        pass

    def seed(self) -> int:
        return abs(hash(self))

    def rng(
        self,
    ) -> Generator:
        return default_rng(self.seed())

    def print_seed(self, seed_queue: Queue):
        seed_queue.put({"Task": str(self), "Seed": self.seed()})


@dataclass(frozen=True)
class ATrainTask(Task):
    name = "A_train"
    m: int
    n: int
    Atr_id: int = field(compare=False)
    Atr_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Atr_seed", seeds.A_train[self.Atr_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_A_train(
            self.m,
            self.n,
            self.Atr_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class ATestTask(Task):
    name = "A_test"
    m: int
    n: int
    Ate_id: int = field(compare=False)
    Ate_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Ate_seed", seeds.A_test[self.Ate_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_A_test(
            self.m,
            self.n,
            self.Ate_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class MoTask(Task):
    name = "Mo"
    m: int
    Mo: ModelType
    ko: int
    Mo_id: int = field(compare=False)
    Mo_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.Mo_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_Mo(
            self.m,
            self.Mo,
            self.ko,
            self.Mo_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class DTask(Task):
    name = "D"
    m: int
    ntr: int
    Atr_id: int = field(compare=False)
    Atr_seed: int = field(init=False)
    Mo: ModelType
    ko: int
    Mo_id: int = field(compare=False)
    Mo_seed: int = field(init=False)
    n: int
    error: float
    D_id: int = field(compare=False)
    D_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Atr_seed", seeds.A_train[self.Atr_id])
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.Mo_id])
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_D(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class MIPTask(Task):
    name = "MIP"
    m: int
    ntr: int
    Atr_id: int = field(compare=False)
    Atr_seed: int = field(init=False)
    Mo: ModelType
    ko: int
    Mo_id: int = field(compare=False)
    Mo_seed: int = field(init=False)
    n: int
    error: float
    D_id: int = field(compare=False)
    D_seed: int = field(init=False)
    ke: int
    config: MIPConfig
    Me_id: int = field(compare=False)
    Me_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Atr_seed", seeds.A_train[self.Atr_id])
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.Mo_id])
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        time, best_fitness = run_MIP(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            self.ke,
            self.config,
            self.Me_id,
            dir,
            self.rng().integers(2_000_000_000),
        )
        queues["train"].put(
            {
                "M": self.m,
                "N_tr": self.ntr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo,
                "Ko": self.ko,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "Me": "SRMP",
                "Ke": self.ke,
                "Method": "MIP",
                "Config": self.config.id,
                "Time": time,
                "Fitness": best_fitness,
            }
        )


@dataclass(frozen=True)
class SATask(Task):
    name = "SA"
    m: int
    ntr: int
    Atr_id: int = field(compare=False)
    Atr_seed: int = field(init=False)
    Mo: ModelType
    ko: int
    Mo_id: int = field(compare=False)
    Mo_seed: int = field(init=False)
    n: int
    error: float
    D_id: int = field(compare=False)
    D_seed: int = field(init=False)
    Me: ModelType
    ke: int
    config: SAConfig
    Me_id: int = field(compare=False)
    Me_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Atr_seed", seeds.A_train[self.Atr_id])
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.Mo_id])
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        time, it, best_fitness = run_SA(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.config,
            self.Me_id,
            dir,
            self.rng(),
        )
        queues["train"].put(
            {
                "M": self.m,
                "N_tr": self.ntr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo,
                "Ko": self.ko,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "Me": self.Me,
                "Ke": self.ke,
                "Method": "SA",
                "Config": self.config.id,
                "Time": time,
                "Fitness": best_fitness,
                "It.": it,
            }
        )


@dataclass(frozen=True)
class TestTask(Task):
    name = "Test"
    m: int
    ntr: int
    Atr_id: int = field(compare=False)
    Atr_seed: int = field(init=False)
    Mo: ModelType
    ko: int
    Mo_id: int = field(compare=False)
    Mo_seed: int = field(init=False)
    n: int
    error: float
    D_id: int = field(compare=False)
    D_seed: int = field(init=False)
    Me: ModelType
    ke: int
    method: Literal["MIP", "SA"]
    config: int
    Me_id: int = field(compare=False)
    Me_seed: int = field(init=False)
    nte: int
    Ate_id: int = field(compare=False)
    Ate_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        object.__setattr__(self, "Atr_seed", seeds.A_train[self.Atr_id])
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.Mo_id])
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])
        object.__setattr__(self, "Ate_seed", seeds.A_test[self.Ate_id])

    def __call__(self, dir, queues):
        test_fitness, kendall_tau = run_test(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.method,
            self.config,
            self.Me_id,
            self.nte,
            self.Ate_id,
            dir,
        )
        queues["test"].put(
            {
                "M": self.m,
                "N_tr": self.ntr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo,
                "Ko": self.ko,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "Me": self.Me,
                "Ke": self.ke,
                "Method": self.method,
                "Config": self.config,
                "N_te": self.nte,
                "Ate_id": self.Ate_id,
                "Fitness": test_fitness,
                "Kendall's tau": kendall_tau,
            }
        )
