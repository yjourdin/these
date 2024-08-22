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
from .fieldnames import DSizeFieldnames, SeedFieldnames, TestFieldnames, TrainFieldnames
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
    ) -> None: ...

    def seed(self) -> int:
        return abs(hash(self))

    def rng(
        self,
    ) -> Generator:
        return default_rng(self.seed())

    def print_seed(self, seed_queue: Queue):
        seed_queue.put({SeedFieldnames.Task: self, SeedFieldnames.Seed: self.seed()})


@dataclass(frozen=True)
class AbstractMTask(Task):
    m: int

    def __post_init__(self, seeds: Seeds): ...


@dataclass(frozen=True)
class ATrainTask(AbstractMTask):
    name = "A_train"
    n_tr: int
    Atr_id: int = field(hash=False)
    Atr_seed: int = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Atr_seed", seeds.A_tr[self.Atr_id])

    def __call__(self, dir):
        self.print_seed(dir.csv_files["seeds"].queue)
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
    Ate_id: int = field(hash=False)
    Ate_seed: int = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Ate_seed", seeds.A_te[self.Ate_id])

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
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
    Mo_id: int = field(hash=False)
    Mo_seed: int = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.group_size][self.Mo_id])

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
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
    same_alt: bool
    error: float
    D_id: int = field(hash=False)
    D_seed: int = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])


@dataclass(frozen=True)
class DTask(AbstractDTask):
    name = "D"
    same_alt: bool = field(default=False, init=False)
    dm_id: int

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
        D_size = create_D(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            False,
            self.error,
            self.dm_id,
            self.D_id,
            dir,
            self.rng(),
        )
        dir.csv_files["D_size"].queue.put(
            {
                DSizeFieldnames.M: self.m,
                DSizeFieldnames.N_tr: self.n_tr,
                DSizeFieldnames.Atr_id: self.Atr_id,
                DSizeFieldnames.Mo: self.Mo,
                DSizeFieldnames.Ko: self.ko,
                DSizeFieldnames.Group_size: self.group_size,
                DSizeFieldnames.Mo_id: self.Mo_id,
                DSizeFieldnames.N_bc: self.n,
                DSizeFieldnames.Same_alt: self.same_alt,
                DSizeFieldnames.Error: self.error,
                DSizeFieldnames.D_id: self.D_id,
                DSizeFieldnames.Size: D_size,
            }
        )


@dataclass(frozen=True)
class DSameTask(AbstractDTask):
    name = "D"
    same_alt: bool = field(default=True, init=False)
    dm_id: int = field(hash=False)

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
        D_size = create_D(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            True,
            self.error,
            self.dm_id,
            self.D_id,
            dir,
            self.rng(),
        )
        dir.csv_files["D_size"].queue.put(
            {
                DSizeFieldnames.M: self.m,
                DSizeFieldnames.N_tr: self.n_tr,
                DSizeFieldnames.Atr_id: self.Atr_id,
                DSizeFieldnames.Mo: self.Mo,
                DSizeFieldnames.Ko: self.ko,
                DSizeFieldnames.Group_size: self.group_size,
                DSizeFieldnames.Mo_id: self.Mo_id,
                DSizeFieldnames.N_bc: self.n,
                DSizeFieldnames.Same_alt: self.same_alt,
                DSizeFieldnames.Error: self.error,
                DSizeFieldnames.D_id: self.D_id,
                DSizeFieldnames.Size: D_size,
            }
        )


@dataclass(frozen=True)
class AbstractElicitationTask(AbstractDTask):
    Me: GroupModelEnum
    ke: int
    method: MethodEnum
    config: Config
    Me_id: int = field(hash=False)
    Me_seed: int = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])


@dataclass(frozen=True)
class MIPTask(AbstractElicitationTask):
    name = "MIP"
    method: MethodEnum = field(default=MethodEnum.MIP, init=False)
    config: MIPConfig

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
        time, best_fitness = run_MIP(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.same_alt,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.config,
            self.Me_id,
            dir,
            self.rng().integers(2_000_000_000),
        )
        dir.csv_files["train"].queue.put(
            {
                TrainFieldnames.M: self.m,
                TrainFieldnames.N_tr: self.n_tr,
                TrainFieldnames.Atr_id: self.Atr_id,
                TrainFieldnames.Mo: self.Mo,
                TrainFieldnames.Ko: self.ko,
                TrainFieldnames.Group_size: self.group_size,
                TrainFieldnames.Mo_id: self.Mo_id,
                TrainFieldnames.N_bc: self.n,
                TrainFieldnames.Same_alt: self.same_alt,
                TrainFieldnames.Error: self.error,
                TrainFieldnames.D_id: self.D_id,
                TrainFieldnames.Me: self.Me,
                TrainFieldnames.Ke: self.ke,
                TrainFieldnames.Method: MethodEnum.MIP,
                TrainFieldnames.Config: self.config,
                TrainFieldnames.Me_id: self.Me_id,
                TrainFieldnames.Time: time,
                TrainFieldnames.Fitness: best_fitness,
            }
        )


@dataclass(frozen=True)
class SATask(AbstractElicitationTask):
    name = "SA"
    method: MethodEnum = field(default=MethodEnum.SA, init=False)
    config: SAConfig

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)
        time, it, best_fitness = run_SA(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.same_alt,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.config,
            self.Me_id,
            dir,
            self.rng(),
        )
        dir.csv_files["train"].queue.put(
            {
                TrainFieldnames.M: self.m,
                TrainFieldnames.N_tr: self.n_tr,
                TrainFieldnames.Atr_id: self.Atr_id,
                TrainFieldnames.Mo: self.Mo,
                TrainFieldnames.Ko: self.ko,
                TrainFieldnames.Group_size: self.group_size,
                TrainFieldnames.Mo_id: self.Mo_id,
                TrainFieldnames.N_bc: self.n,
                TrainFieldnames.Same_alt: self.same_alt,
                TrainFieldnames.Error: self.error,
                TrainFieldnames.D_id: self.D_id,
                TrainFieldnames.Me: self.Me,
                TrainFieldnames.Ke: self.ke,
                TrainFieldnames.Method: MethodEnum.SA,
                TrainFieldnames.Config: self.config,
                TrainFieldnames.Me_id: self.Me_id,
                TrainFieldnames.Time: time,
                TrainFieldnames.Fitness: best_fitness,
                TrainFieldnames.It: it,
            }
        )


@dataclass(frozen=True)
class TestTask(ATestTask, AbstractElicitationTask):
    name = "Test"

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)

    def __call__(self, dir: Directory):
        test_fitness, kendall_tau = run_test(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.n,
            self.same_alt,
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
        dir.csv_files["test"].queue.put(
            {
                TestFieldnames.M: self.m,
                TestFieldnames.N_tr: self.n_tr,
                TestFieldnames.Atr_id: self.Atr_id,
                TestFieldnames.Mo: self.Mo,
                TestFieldnames.Ko: self.ko,
                TestFieldnames.Group_size: self.group_size,
                TestFieldnames.Mo_id: self.Mo_id,
                TestFieldnames.N_bc: self.n,
                TestFieldnames.Same_alt: self.same_alt,
                TestFieldnames.Error: self.error,
                TestFieldnames.D_id: self.D_id,
                TestFieldnames.Me: self.Me,
                TestFieldnames.Ke: self.ke,
                TestFieldnames.Method: self.method,
                TestFieldnames.Config: self.config,
                TestFieldnames.Me_id: self.Me_id,
                TestFieldnames.N_te: self.n_te,
                TestFieldnames.Ate_id: self.Ate_id,
                TestFieldnames.Fitness: test_fitness,
                TestFieldnames.Kendall: kendall_tau,
            }
        )
