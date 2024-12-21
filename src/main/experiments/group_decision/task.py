from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from typing import Any, ClassVar

from pandas import read_csv

from ..dataclass import FrozenDataclass
from ..methods import MethodEnum
from ..mip.main import learn_mip
from ..model import GroupModel
from ..models import GroupModelEnum, model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.generate import noisy_comparisons, random_comparisons
from ..preference_structure.io import from_csv, to_csv
from ....random import Seed, SeedMixin, seed
from ..sa.main import learn_sa
from ..test.main import test_consensus, test_distance
from ..test.test import DistanceRankingEnum
from .experiments.elicitation.config import Config, MIPConfig, SAConfig, SRMPSAConfig
from .directory import Directory
from .fieldnames import DSizeFieldnames, SeedFieldnames, TestFieldnames, TrainFieldnames
from .experiments.elicitation.seeds import Seeds


@dataclass(frozen=True)
class Task(FrozenDataclass, SeedMixin):
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

    @property
    def seed(self) -> Seed:
        return abs(hash(self))

    def print_seed(self, seed_queue: Queue):
        seed_queue.put({SeedFieldnames.Task: self, SeedFieldnames.Seed: self.seed})

    @abstractmethod
    def already_done(self, dir: Directory): ...


@dataclass(frozen=True)
class AbstractMTask(Task):
    m: int


@dataclass(frozen=True)
class ATrainTask(AbstractMTask):
    name = "A_train"
    ntr: int
    Atr_id: int

    def __call__(self, A_train_file: Path, seed: Seed):
        self.print_seed(dir.csv_files["seeds"].queue)

        A = NormalPerformanceTable.random(self.ntr, self.m, self.rng)

        with A_train_file.open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def already_done(self, A_train_file: Path, seed: Seed):
        return A_train_file.exists()


@dataclass(frozen=True)
class ATestTask(AbstractMTask):
    name = "A_test"
    nte: int
    Ate_id: int = field(hash=False)
    Ate_seed: Seed = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Ate_seed", seeds.A_te[self.Ate_id])

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)

        A = NormalPerformanceTable.random(self.nte, self.m, self.rng)

        with self.A_test_file(dir).open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def A_test_file(self, dir: Directory):
        return dir.A_test(self.m, self.nte, self.Ate_id)

    def already_done(self, dir: Directory):
        return self.A_test_file(dir).exists()


@dataclass(frozen=True)
class MoTask(AbstractMTask):
    name = "Mo"
    Mo: GroupModelEnum
    ko: int
    group_size: int
    Mo_id: int = field(hash=False)
    Mo_seed: Seed = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Mo_seed", seeds.Mo[self.group_size][self.Mo_id])

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)

        Mo = model(*self.Mo.value, self.group_size).random(
            nb_profiles=self.ko,
            nb_crit=self.m,
            rng=self.rng,
            **({"group_size": self.group_size} if self.group_size > 1 else {}),
        )

        with self.Mo_file(dir).open("w") as f:
            f.write(Mo.to_json())

    def Mo_file(self, dir: Directory):
        return dir.Mo(self.m, self.Mo, self.ko, self.group_size, self.Mo_id)

    def already_done(self, dir: Directory):
        return self.Mo_file(dir).exists()


@dataclass(frozen=True)
class AbstractDTask(MoTask, ATrainTask):
    nbc: int
    same_alt: bool
    error: float
    D_id: int = field(hash=False)
    D_seed: Seed = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "D_seed", seeds.D[self.D_id])

    def D_file(self, dir: Directory, dm_id: int):
        return dir.D(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.nbc,
            self.same_alt,
            self.error,
            dm_id,
            self.D_id,
        )


@dataclass(frozen=True)
class DTask(AbstractDTask):
    name = "D"
    dm_id: int

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)

        with self.Mo_file(dir).open("r") as f:
            Mo = model(*self.Mo.value, self.group_size).from_json(f.read())

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        rng_shuffle, rng_error = self.rng.spawn(2)

        D = random_comparisons(
            A,
            Mo[self.dm_id] if isinstance(Mo, GroupModel) else Mo,
            self.nbc,
            rng_shuffle,
        )

        if self.error:
            D = noisy_comparisons(D, self.error, rng_error)

        with self.D_file(dir, self.dm_id).open("w") as f:
            to_csv(D, f)

        dir.csv_files["D_size"].queue.put(
            {
                DSizeFieldnames.M: self.m,
                DSizeFieldnames.N_tr: self.ntr,
                DSizeFieldnames.Atr_id: self.Atr_id,
                DSizeFieldnames.Mo: self.Mo,
                DSizeFieldnames.Ko: self.ko,
                DSizeFieldnames.Group_size: self.group_size,
                DSizeFieldnames.Mo_id: self.Mo_id,
                DSizeFieldnames.N_bc: self.nbc,
                DSizeFieldnames.Same_alt: self.same_alt,
                DSizeFieldnames.Error: self.error,
                DSizeFieldnames.D_id: self.D_id,
                DSizeFieldnames.Size: len(D),
            }
        )

    def already_done(self, dir: Directory):
        return self.D_file(dir, self.dm_id).exists()


@dataclass(frozen=True)
class AbstractElicitationTask(AbstractDTask):
    Me: GroupModelEnum
    ke: int
    method: MethodEnum
    config: Config
    Me_id: int = field(hash=False)
    Me_seed: Seed = field(init=False, hash=True, compare=False)

    def __post_init__(self, seeds: Seeds):
        super().__post_init__(seeds)
        object.__setattr__(self, "Me_seed", seeds.Me[self.Me_id])

    def Me_file(self, dir: Directory):
        return dir.Me(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo,
            self.ko,
            self.group_size,
            self.Mo_id,
            self.nbc,
            self.same_alt,
            self.error,
            self.D_id,
            self.Me,
            self.ke,
            self.method,
            self.config,
            self.Me_id,
        )

    def already_done(self, dir: Directory):
        return self.Me_file(dir).exists()


@dataclass(frozen=True)
class MIPTask(AbstractElicitationTask):
    name = "MIP"
    method: MethodEnum = field(default=MethodEnum.MIP, init=False)
    config: MIPConfig

    def __call__(self, dir: Directory):
        self.print_seed(dir.csv_files["seeds"].queue)

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D = []
        for dm_id in range(self.group_size):
            with self.D_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        best_model, best_fitness, time = learn_mip(
            self.Me,
            self.ke,
            A,
            D,
            self.config.max_time,
            seed=seed(self.rng, 2_000_000_000),
            gamma=self.config.gamma,
        )

        with self.Me_file(dir).open("w") as f:
            f.write(best_model.to_json() if best_model else "None")

        dir.csv_files["train"].queue.put(
            {
                TrainFieldnames.M: self.m,
                TrainFieldnames.N_tr: self.ntr,
                TrainFieldnames.Atr_id: self.Atr_id,
                TrainFieldnames.Mo: self.Mo,
                TrainFieldnames.Ko: self.ko,
                TrainFieldnames.Group_size: self.group_size,
                TrainFieldnames.Mo_id: self.Mo_id,
                TrainFieldnames.N_bc: self.nbc,
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

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D = []
        for dm_id in range(self.group_size):
            with self.D_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        rng_init, rng_sa = self.rng.spawn(2)

        best_model, best_fitness, time, it = learn_sa(
            self.Me.value[0],
            self.ke,
            A,
            D[0],
            self.config.alpha,
            rng_init,
            rng_sa,
            accept=self.config.accept,
            max_time=self.config.max_time,
            max_it=self.config.max_it,
            **(
                {"amp": self.config.amp}
                if isinstance(self.config, SRMPSAConfig)
                else {}  # type: ignore
            ),
        )

        with self.Me_file(dir).open("w") as f:
            f.write(best_model.to_json())

        dir.csv_files["train"].queue.put(
            {
                TrainFieldnames.M: self.m,
                TrainFieldnames.N_tr: self.ntr,
                TrainFieldnames.Atr_id: self.Atr_id,
                TrainFieldnames.Mo: self.Mo,
                TrainFieldnames.Ko: self.ko,
                TrainFieldnames.Group_size: self.group_size,
                TrainFieldnames.Mo_id: self.Mo_id,
                TrainFieldnames.N_bc: self.nbc,
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
        csv_fields = {
            TestFieldnames.M: self.m,
            TestFieldnames.N_tr: self.ntr,
            TestFieldnames.Atr_id: self.Atr_id,
            TestFieldnames.Mo: self.Mo,
            TestFieldnames.Ko: self.ko,
            TestFieldnames.Group_size: self.group_size,
            TestFieldnames.Mo_id: self.Mo_id,
            TestFieldnames.N_bc: self.nbc,
            TestFieldnames.Same_alt: self.same_alt,
            TestFieldnames.Error: self.error,
            TestFieldnames.D_id: self.D_id,
            TestFieldnames.Me: self.Me,
            TestFieldnames.Ke: self.ke,
            TestFieldnames.Method: self.method,
            TestFieldnames.Config: self.config,
            TestFieldnames.Me_id: self.Me_id,
            TestFieldnames.N_te: self.nte,
            TestFieldnames.Ate_id: self.Ate_id,
        }

        with self.A_test_file(dir).open("r") as f:
            A_test = NormalPerformanceTable(read_csv(f, header=None))

        with self.Mo_file(dir).open("r") as f:
            Mo = model(*self.Mo.value, self.group_size).from_json(f.read())

        with self.Me_file(dir).open("r") as f:
            s = f.read()
            try:
                Me = model(*self.Me.value, self.group_size).from_json(s)
            except ValueError:
                Me = None

        def put_in_queue(name, value):
            dir.csv_files["test"].queue.put(
                csv_fields | {TestFieldnames.Name: name, TestFieldnames.Value: value}
            )

        def write_consensus(model: GroupModel, prefix: str = ""):
            for name, value in test_consensus(model, A_test, distance):
                put_in_queue("_".join([prefix, str(distance), name]), value)

        for distance in DistanceRankingEnum:
            if isinstance(Mo, GroupModel):
                write_consensus(Mo, "Mo")
            if Me:
                if isinstance(Me, GroupModel):
                    write_consensus(Me, "Me")
                for name, value in test_distance(Mo, Me, A_test, distance):
                    put_in_queue(name, value)

    def already_done(self, dir: Directory):
        return False
