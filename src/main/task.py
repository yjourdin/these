from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from enum import Enum
from multiprocessing import Queue
from typing import Any, ClassVar

from numpy.random import Generator, default_rng

from ..dataclass import FrozenDataclass
from ..methods import MethodEnum
from ..models import GroupModelEnum
from .config import Config, MIPConfig, SAConfig
from .directory import Directory
from .job import (
    create_A_test,
    create_A_train,
    create_D,
    create_Mo,
    run_MIP,
    run_SA,
    run_test,
)
from .seeds import Seeds


def print_attribute(name: str, value: Any | None, width: int) -> str:
    if value is not None:
        if isinstance(value, Enum):
            value = value.name
        if isinstance(value, Config):
            value = value.id
        return f"{name.capitalize()}: {value:{width}}"
    else:
        return f"{'':{len(name)}}  {'':{width}}"


WIDTH = {
    "m": 2,
    "n_tr": 4,
    "Atr_id": 2,
    "Mo": 8,
    "ko": 2,
    "group_size": 3,
    "Mo_id": 2,
    "n": 4,
    "error": 4,
    "dm_id": 3,
    "D_id": 2,
    "Me": 8,
    "ke": 2,
    "method": 3,
    "config": 4,
    "Me_id": 2,
    "n_te": 4,
    "Ate_id": 2,
}


@dataclass(frozen=True)
class Task(FrozenDataclass):
    name: ClassVar[str]
    seeds: InitVar[Seeds]

    @abstractmethod
    def __post_init__(self, seeds: Seeds): ...

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
        return abs(hash((v for k, v in self.to_dict().items() if not k.endswith("id"))))

    def rng(
        self,
    ) -> Generator:
        return default_rng(self.seed())

    def print_seed(self, seed_queue: Queue):
        seed_queue.put({"Task": str(self), "Seed": self.seed()})


@dataclass(frozen=True)
class AbstractMTask(Task):
    m: int

    def __post_init__(self, seeds: Seeds):
        return ...


@dataclass(frozen=True)
class ATrainTask(AbstractMTask):
    name = "A_train"
    n_tr: int
    Atr_id: int
    Atr_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Atr_seed", seeds.A_tr[self.Atr_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_A_train(
            self.m,
            self.n_tr,
            self.Atr_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class ATestTask(AbstractMTask):
    name = "A_test"
    n_te: int
    Ate_id: int
    Ate_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Ate_seed", seeds.A_te[self.Ate_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_A_test(
            self.m,
            self.n_te,
            self.Ate_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class MoTask(AbstractMTask):
    name = "Mo"
    Mo: GroupModelEnum
    ko: int
    group_size: int
    Mo_id: int
    Mo_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.group_size][self.Mo_id])

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_Mo(
            self.m,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class AbstractDTask(MoTask, ATrainTask):
    n: int
    error: float
    D_id: int
    D_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])


@dataclass(frozen=True)
class DTask(AbstractDTask):
    name = "D"
    dm_id: int

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        create_D(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.error,
            self.dm_id,
            self.D_id,
            dir,
            self.rng(),
        )


@dataclass(frozen=True)
class AbstractElicitationTask(AbstractDTask):
    Me: GroupModelEnum
    ke: int
    Me_id: int
    Me_seed: int = field(init=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])


@dataclass(frozen=True)
class MIPTask(AbstractElicitationTask):
    name = "MIP"
    config: MIPConfig

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        time, best_fitness = run_MIP(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.config,
            self.Me_id,
            dir,
            self.rng().integers(2_000_000_000),
        )
        queues["train"].put(
            {
                "M": self.m,
                "N_tr": self.n_tr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo.name,
                "Ko": self.ko,
                "Group_size": self.group_size,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "D_id": self.D_id,
                "Me": self.Me.name,
                "Ke": self.ke,
                "Method": MethodEnum.MIP.name,
                "Config": self.config.id,
                "Me_id": self.Me_id,
                "Time": time,
                "Fitness": best_fitness,
            }
        )


@dataclass(frozen=True)
class SATask(AbstractElicitationTask):
    name = "SA"
    config: SAConfig

    def __call__(self, dir, queues):
        self.print_seed(queues["seeds"])
        time, it, best_fitness = run_SA(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
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
                "N_tr": self.n_tr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo.name,
                "Ko": self.ko,
                "Group_size": self.group_size,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "D_id": self.D_id,
                "Me": self.Me.name,
                "Ke": self.ke,
                "Method": MethodEnum.SA.name,
                "Config": self.config.id,
                "Me_id": self.Me_id,
                "Time": time,
                "Fitness": best_fitness,
                "It.": it,
            }
        )


@dataclass(frozen=True)
class TestTask(ATestTask, AbstractElicitationTask):
    name = "Test"
    config: int
    method: MethodEnum

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)

    def __call__(self, dir, queues):
        test_fitness, kendall_tau = run_test(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.method,
            self.config,
            self.Me_id,
            self.n_te,
            self.Ate_id,
            dir,
        )
        queues["test"].put(
            {
                "M": self.m,
                "N_tr": self.n_tr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo.name,
                "Ko": self.ko,
                "Group_size": self.group_size,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "D_id": self.D_id,
                "Me": self.Me.name,
                "Ke": self.ke,
                "Method": self.method.name,
                "Config": self.config,
                "Me_id": self.Me_id,
                "N_te": self.n_te,
                "Ate_id": self.Ate_id,
                "Fitness": test_fitness,
                "Kendall's tau": kendall_tau,
            }
        )
