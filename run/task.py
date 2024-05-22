from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Any, ClassVar, Literal, cast

from numpy.random import default_rng

from model import ModelType

from .arguments import ConfigDict
from .config import SAConfig
from .job import (
    create_A_test,
    create_A_train,
    create_D,
    create_Mo,
    run_MIP,
    run_SA,
    run_test,
)
from .path import Directory
from .seed import Seeds, seed


def print_attribute(name: str, value: Any | None, width: int) -> str:
    if value is not None:
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
    "Me": 4,
    "ke": 2,
    "config": 4,
    "method": 3,
    "n_te": 4,
    "Ate_id": 2,
}


@dataclass(frozen=True)
class Task(ABC):
    name: ClassVar[str]

    def __str__(self) -> str:
        attributes = (
            print_attribute(attr, getattr(self, attr, None), width)
            for attr, width in WIDTH.items()
        )
        return f"{self.name:7} ({', '.join(attributes)})"

    @abstractmethod
    def __call__(
        self,
        configs: ConfigDict,
        dir: Directory,
        seeds: Seeds,
        queues: dict[str, Queue],
    ) -> None:
        pass


@dataclass(frozen=True)
class ATrainTask(Task):
    name = "A_train"
    m: int
    n: int
    Atr_id: int

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        create_A_train(
            self.m,
            self.n,
            self.Atr_id,
            dir,
            default_rng(seed),
        )

    def seed(self, seeds):
        return seed(default_rng([self.n, self.m, seeds.A_train[self.Atr_id]]))


@dataclass(frozen=True)
class ATestTask(Task):
    name = "A_test"
    m: int
    n: int
    Ate_id: int

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        create_A_test(
            self.m,
            self.n,
            self.Ate_id,
            dir,
            default_rng(seed),
        )

    def seed(self, seeds: Seeds):
        return seed(default_rng([self.n, self.m, seeds.A_test[self.Ate_id]]))


@dataclass(frozen=True)
class MoTask(Task):
    name = "Mo"
    m: int
    Mo: ModelType
    ko: int
    Mo_id: int

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        create_Mo(
            self.m,
            self.Mo,
            self.ko,
            self.Mo_id,
            dir,
            default_rng(seed),
        )

    def seed(self, seeds: Seeds):
        return seed(default_rng([self.ko, self.m, seeds.Mo[self.Mo_id]]))


@dataclass(frozen=True)
class DTask(Task):
    name = "D"
    m: int
    n_tr: int
    Atr_id: int
    Mo: ModelType
    ko: int
    Mo_id: int
    n: int
    error: float

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        create_D(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            dir,
            default_rng(seed),
        )

    def seed(self, seeds: Seeds):
        return seed(
            default_rng(
                [
                    self.n,
                    seeds.Mo[self.Mo_id],
                    self.ko,
                    seeds.A_train[self.Atr_id],
                    self.n_tr,
                    self.m,
                ]
            )
        )


@dataclass(frozen=True)
class SATask(Task):
    name = "SA"
    m: int
    n_tr: int
    Atr_id: int
    Mo: ModelType
    ko: int
    Mo_id: int
    n: int
    error: float
    Me: ModelType
    ke: int
    config: int

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        time, it, best_fitness = run_SA(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.Me,
            self.ke,
            self.config,
            cast(SAConfig, configs["SA"][self.config]),
            dir,
            default_rng(seed),
        )
        queues["train"].put(
            {
                "M": self.m,
                "N_tr": self.n_tr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo,
                "Ko": self.ko,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "Me": self.Me,
                "Ke": self.ke,
                "Method": "SA",
                "Config": self.config,
                "Time": time,
                "Fitness": best_fitness,
                "It.": it,
            }
        )

    def seed(self, seeds: Seeds):
        return seed(
            default_rng(
                [
                    self.ke,
                    self.n,
                    seeds.Mo[self.Mo_id],
                    self.ko,
                    seeds.A_train[self.Atr_id],
                    self.n_tr,
                    self.m,
                ]
            )
        )


@dataclass(frozen=True)
class MIPTask(Task):
    name = "MIP"
    m: int
    n_tr: int
    Atr_id: int
    Mo: ModelType
    ko: int
    Mo_id: int
    n: int
    error: float
    ke: int

    def __call__(self, configs, dir, seeds, queues):
        seed = self.seed(seeds)
        queues["seeds"].put({"Task": str(self), "Seed": seed})

        time, best_fitness = run_MIP(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.ke,
            dir,
            default_rng(seed).integers(2_000_000_000),
        )
        queues["train"].put(
            {
                "M": self.m,
                "N_tr": self.n_tr,
                "Atr_id": self.Atr_id,
                "Mo": self.Mo,
                "Ko": self.ko,
                "Mo_id": self.Mo_id,
                "N_bc": self.n,
                "Error": self.error,
                "Me": "SRMP",
                "Ke": self.ke,
                "Method": "MIP",
                "Time": time,
                "Fitness": best_fitness,
            }
        )

    def seed(self, seeds: Seeds):
        return seed(
            default_rng(
                [
                    self.ke,
                    self.n,
                    seeds.Mo[self.Mo_id],
                    self.ko,
                    seeds.A_train[self.Atr_id],
                    self.n_tr,
                    self.m,
                ]
            )
        )


@dataclass(frozen=True)
class TestTask(Task):
    name = "Test"
    m: int
    n_tr: int
    Atr_id: int
    Mo: ModelType
    ko: int
    Mo_id: int
    n: int
    error: float
    Me: ModelType
    ke: int
    method: Literal["MIP", "SA"]
    config: int
    n_te: int
    Ate_id: int

    def __call__(self, configs, dir, seeds, queues):
        test_fitness, kendall_tau = run_test(
            self.m,
            self.n_tr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.Mo_id,
            self.n,
            self.error,
            self.Me,
            self.ke,
            self.method,
            self.config,
            self.n_te,
            self.Ate_id,
            dir,
        )
        queues["test"].put(
            {
                "M": self.m,
                "N_tr": self.n_tr,
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
                "N_te": self.n_te,
                "Ate_id": self.Ate_id,
                "Fitness": test_fitness,
                "Kendall's tau": kendall_tau,
            }
        )
