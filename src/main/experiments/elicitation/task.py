from dataclasses import dataclass, field, replace
from typing import Any, cast

from mcda.relations import PreferenceStructure
from pandas import read_csv

from ....methods import MethodEnum
from ....mip.main import learn_mip
from ....model import GroupModel, Model
from ....models import GroupModelEnum, model
from ....performance_table.normal_performance_table import NormalPerformanceTable
from ....preference_structure.generate import noisy_comparisons, random_comparisons
from ....preference_structure.io import from_csv, to_csv
from ....random import SeedLike
from ....random import seed as random_seed
from ....sa.main import learn_sa
from ....test.main import test_consensus, test_distance
from ....test.test import DistanceRankingEnum
from ....utils import tolist
from ...task import SeedTask
from .config import Config, MIPConfig, SAConfig
from .directory import DirectoryElicitation


@dataclass(frozen=True)
class AbstractMTask(SeedTask):
    m: int


@dataclass(frozen=True)
class ATrainTask(AbstractMTask):
    name = "A_train"
    ntr: int
    Atr_id: int = field(hash=False)

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        A = NormalPerformanceTable.random(self.ntr, self.m, self.rng(seed))

        with self.A_train_file(dir).open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def A_train_file(self, dir: DirectoryElicitation):
        return dir.A_train(self.m, self.ntr, self.Atr_id)

    def done(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        return self.A_train_file(dir).exists()


@dataclass(frozen=True)
class ATestTask(AbstractMTask):
    name = "A_test"
    nte: int
    Ate_id: int = field(hash=False)

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        A = NormalPerformanceTable.random(self.nte, self.m, self.rng(seed))

        with self.A_test_file(dir).open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def A_test_file(self, dir: DirectoryElicitation):
        return dir.A_test(self.m, self.nte, self.Ate_id)

    def done(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        return self.A_test_file(dir).exists()


@dataclass(frozen=True)
class MoTask(AbstractMTask):
    name = "Mo"
    Mo: GroupModelEnum
    ko: int
    group_size: int
    fixed_lex_order: bool = field(hash=False)
    Mo_id: int = field(hash=False)

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        Mo = model(self.Mo, self.group_size).random(
            nb_profiles=self.ko,
            nb_crit=self.m,
            rng=self.rng(seed),
            **({"group_size": self.group_size} if self.group_size > 1 else {}),
        )

        with self.Mo_file(dir).open("w") as f:
            f.write(Mo.to_json())

    @property
    def lexicographic_order(self) -> list[int]:
        return tolist(
            MoTask(self.m, self.Mo, self.ko, self.group_size, True, self.Mo_id)
            .rng(self.Mo_id)
            .permutation(self.ko)
        )

    def Mo_file(self, dir: DirectoryElicitation):
        return dir.Mo(self.m, self.Mo, self.ko, self.group_size, self.Mo_id)

    def done(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        return self.Mo_file(dir).exists()


@dataclass(frozen=True)
class AbstractDTask(MoTask, ATrainTask):
    nbc: int
    same_alt: bool
    error: float
    D_id: int = field(hash=False)

    def D_file(self, dir: DirectoryElicitation, dm_id: int):
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

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        with self.Mo_file(dir).open("r") as f:
            Mo = model(self.Mo, self.group_size).from_json(f.read())

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        rng_shuffle, rng_error = self.rng(seed).spawn(2)
        if self.same_alt:
            rng_shuffle = replace(self, dm_id=0).rng(seed)

        d = random_comparisons(
            A,
            cast(Model, Mo[self.dm_id]) if isinstance(Mo, GroupModel) else Mo,
            self.nbc,
            rng=rng_shuffle,
        )

        if self.error:
            d = noisy_comparisons(d, self.error, rng_error)

        with self.D_file(dir, self.dm_id).open("w") as f:
            to_csv(d, f)

    def done(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        return self.D_file(dir, self.dm_id).exists()


@dataclass(frozen=True)
class AbstractElicitationTask(AbstractDTask):
    Me: GroupModelEnum
    ke: int
    method: MethodEnum
    config: Config
    Me_id: int = field(hash=False)

    def Me_file(self, dir: DirectoryElicitation):
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

    def done(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        return self.Me_file(dir).exists()


@dataclass(frozen=True)
class MIPTask(AbstractElicitationTask):
    name = "MIP"
    method: MethodEnum = field(default=MethodEnum.MIP, init=False)
    config: MIPConfig

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            with self.D_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        rng_lex, rng_mip = self.rng(seed).spawn(2)

        best_model, best_fitness, time = learn_mip(
            self.Me,
            self.ke,
            A,
            D,
            rng_lex,
            random_seed(rng_mip),
            self.config.max_time,
            self.lexicographic_order if self.fixed_lex_order else None,
            gamma=self.config.gamma,
        )

        with self.Me_file(dir).open("w") as f:
            f.write(best_model.to_json() if best_model else "None")

        csv_file = dir.csv_files["train"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Mo=self.Mo,
                Ko=self.ko,
                Group_size=self.group_size,
                Mo_id=self.Mo_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                Error=self.error,
                D_id=self.D_id,
                Me=self.Me,
                Ke=self.ke,
                Method=MethodEnum.MIP,
                Config=self.config,
                Me_id=self.Me_id,
                Time=time,
                Fitness=best_fitness,
            )
        )


@dataclass(frozen=True)
class SATask(AbstractElicitationTask):
    name = "SA"
    method: MethodEnum = field(default=MethodEnum.SA, init=False)
    config: SAConfig

    def task(
        self, dir: DirectoryElicitation, seed: SeedLike, *args: Any, **kwargs: Any
    ):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            with self.D_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        rng_init, rng_sa = self.rng(seed).spawn(2)

        best_model, best_fitness, time, it = learn_sa(
            self.Me.value[0],
            self.ke,
            A,
            D[0],
            self.config.alpha,
            self.config.amp,
            rng_init,
            rng_sa,
            self.lexicographic_order if self.fixed_lex_order else None,
            accept=self.config.accept,
            max_time=self.config.max_time,
            max_it=self.config.max_it,
            max_it_non_improving=self.config.max_it_non_improving,
        )

        with self.Me_file(dir).open("w") as f:
            f.write(best_model.to_json())

        csv_file = dir.csv_files["train"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Mo=self.Mo,
                Ko=self.ko,
                Group_size=self.group_size,
                Mo_id=self.Mo_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                Error=self.error,
                D_id=self.D_id,
                Me=self.Me,
                Ke=self.ke,
                Method=MethodEnum.SA,
                Config=self.config,
                Me_id=self.Me_id,
                Time=time,
                It=it,
                Fitness=best_fitness,
            )
        )


@dataclass(frozen=True)
class TestTask(ATestTask, AbstractElicitationTask):
    name = "Test"

    def task(self, dir: DirectoryElicitation, *args: Any, **kwargs: Any):
        with self.A_test_file(dir).open("r") as f:
            A_test = NormalPerformanceTable(read_csv(f, header=None))

        with self.Mo_file(dir).open("r") as f:
            Mo = model(self.Mo, self.group_size).from_json(f.read())

        with self.Me_file(dir).open("r") as f:
            s = f.read()
            try:
                Me = model(self.Me, self.group_size).from_json(s)
            except ValueError:
                Me = None

        def put_in_queue(name: str, value: float):
            csv_file = dir.csv_files["test"]
            csv_file.writerow(
                csv_file.fields(
                    M=self.m,
                    N_tr=self.ntr,
                    Atr_id=self.Atr_id,
                    Mo=self.Mo,
                    Ko=self.ko,
                    Group_size=self.group_size,
                    Mo_id=self.Mo_id,
                    N_bc=self.nbc,
                    Same_alt=self.same_alt,
                    Error=self.error,
                    D_id=self.D_id,
                    Me=self.Me,
                    Ke=self.ke,
                    Method=self.method,
                    Config=self.config,
                    Me_id=self.Me_id,
                    N_te=self.nte,
                    Ate_id=self.Ate_id,
                    Name=name,
                    Value=value,
                )
            )

        def write_consensus(model: GroupModel[Model], prefix: str = ""):
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

    def done(self, *args: Any, **kwargs: Any):
        return False
