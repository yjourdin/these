import csv
import time
from dataclasses import dataclass, field, replace
from typing import cast

from mcda.relations import PreferenceStructure
from pandas import read_csv

from ....mip.main import learn_mip
from ....models import GroupModelEnum, model_from_json
from ....performance_table.normal_performance_table import NormalPerformanceTable
from ....preference_path.main import compute_preference_path
from ....preference_structure.generate import (
    random_comparisons,
)
from ....preference_structure.io import from_csv, to_csv
from ....random import Seed, rng
from ....random import seed as random_seed
from ....srmp.model import SRMPModel
from ...task import SeedTask
from ..elicitation.config import MIPConfig
from .directory import DirectoryGroupDecision
from .fieldnames import CollectiveFieldnames, PathFieldnames
from .hyperparameters import AcceptHyperparameters, GenHyperparameters


@dataclass(frozen=True)
class AbstractMTask(SeedTask):
    m: int


@dataclass(frozen=True)
class ATrainTask(AbstractMTask):
    name = "A_train"
    ntr: int
    Atr_id: int = field(hash=False)

    def task(self, dir: DirectoryGroupDecision, seed: Seed):
        A = NormalPerformanceTable.random(self.ntr, self.m, self.rng(seed))

        with self.A_train_file(dir).open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def A_train_file(self, dir: DirectoryGroupDecision):
        return dir.A_train(self.m, self.ntr, self.Atr_id)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.A_train_file(dir).exists()


@dataclass(frozen=True)
class MoTask(AbstractMTask):
    name = "Mo"
    ko: int
    Mo_id: int = field(hash=False)

    def task(self, dir: DirectoryGroupDecision, seed: Seed):
        Mo = SRMPModel.random(nb_profiles=self.ko, nb_crit=self.m, rng=self.rng(seed))

        with self.Mo_file(dir).open("w") as f:
            f.write(Mo.to_json())

    def Mo_file(self, dir: DirectoryGroupDecision):
        return dir.Mo(self.m, self.ko, self.Mo_id)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.Mo_file(dir).exists()


@dataclass(frozen=True)
class AbstractMiTask(MoTask):
    group_size: int
    gen: GenHyperparameters
    Mi_id: int = field(hash=False)

    def Mi_file(self, dir: DirectoryGroupDecision, dm_id: int):
        return dir.Mi(
            self.m,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            dm_id,
            self.Mi_id,
        )


@dataclass(frozen=True)
class MiTask(AbstractMiTask):
    name = "Mi"
    dm_id: int

    def task(self, dir: DirectoryGroupDecision, seed: Seed):
        with self.Mo_file(dir).open("r") as f:
            Mo = SRMPModel.from_json(f.read())

        Mi = SRMPModel.from_reference(
            Mo, self.gen.P, self.gen.W, self.gen.L, rng=self.rng(seed)
        )

        with self.Mi_file(dir, self.dm_id).open("w") as f:
            f.write(Mi.to_json())

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.Mi_file(dir, self.dm_id).exists()


@dataclass(frozen=True)
class AbstractDTask(AbstractMiTask, ATrainTask):
    nbc: int
    same_alt: bool
    D_id: int = field(hash=False)

    def D_file(self, dir: DirectoryGroupDecision, dm_id: int, it: int):
        return dir.D(
            self.m,
            self.ntr,
            self.Atr_id,
            self.Mo_id,
            self.ko,
            self.group_size,
            self.gen,
            dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            it,
        )


@dataclass(frozen=True)
class DTask(AbstractDTask, MiTask):
    name = "D"

    def task(self, dir: DirectoryGroupDecision, seed: Seed):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.Mi_file(dir, self.dm_id).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        if self.same_alt:
            rng = replace(self, dm_id=0).rng(seed)
        else:
            rng = self.rng(seed)

        D = random_comparisons(A, Mi, self.nbc, rng)

        with self.D_file(dir, self.dm_id, 0).open("w") as f:
            to_csv(D, f)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.D_file(dir, self.dm_id, 0).exists()


@dataclass(frozen=True)
class CollectiveTask(AbstractDTask):
    name = "Collective"
    config: MIPConfig
    Mc_id: int = field(hash=False)
    it: int# = field(hash=False)

    def task(
        self,
        dir: DirectoryGroupDecision,
        seed: Seed,
    ):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            with self.D_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        C: list[int] = []
        with self.C_file(dir).open("r", newline="") as f:
            C_reader = csv.reader(f, dialect="unix")
            for changes in C_reader:
                C.append(int(changes[0]))

        R: list[list[PreferenceStructure]] = []
        for dm_id in range(self.group_size):
            R_dm: list[PreferenceStructure] = []
            for R_file in self.R_dir(dir, dm_id).iterdir():
                with R_file.open("r") as f:
                    R_dm.append(from_csv(f))
            R.append(R_dm)

        rng_lex, rng_mip = self.rng(seed).spawn(2)

        best_model, best_fitness, time = learn_mip(
            GroupModelEnum.SRMP,
            self.ko,
            A,
            D,
            rng_lex,
            random_seed(rng_mip),
            self.config.max_time,
            True,
            C,
            R,
            gamma=self.config.gamma,
        )

        with self.Mc_file(dir).open("w") as f:
            f.write(best_model.to_json() if best_model else "None")

        dir.csv_files["collective"].queue.put(
            {
                CollectiveFieldnames.M: self.m,
                CollectiveFieldnames.N_tr: self.ntr,
                CollectiveFieldnames.Atr_id: self.Atr_id,
                CollectiveFieldnames.Ko: self.ko,
                CollectiveFieldnames.Mo_id: self.Mo_id,
                CollectiveFieldnames.Group_size: self.group_size,
                CollectiveFieldnames.Gen: self.gen,
                CollectiveFieldnames.Mi_id: self.Mi_id,
                CollectiveFieldnames.N_bc: self.nbc,
                CollectiveFieldnames.Same_alt: self.same_alt,
                CollectiveFieldnames.D_id: self.D_id,
                CollectiveFieldnames.Config: self.config,
                CollectiveFieldnames.It: self.it,
                CollectiveFieldnames.Time: time,
                CollectiveFieldnames.Fitness: best_fitness,
            }
        )

    def C_file(self, dir: DirectoryGroupDecision):
        return dir.C(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.it,
        )

    def R_dir(self, dir: DirectoryGroupDecision, dm_id: int):
        return dir.R_dir(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
        )

    def R_file(self, dir: DirectoryGroupDecision, dm_id: int):
        return dir.R_file(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.it,
        )

    def Mc_file(self, dir: DirectoryGroupDecision):
        return dir.Mc(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.it,
        )

    def D_file(self, dir: DirectoryGroupDecision, dm_id: Seed):
        return super().D_file(dir, dm_id, self.it)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.Mc_file(dir).exists()


@dataclass(frozen=True)
class PreferencePathTask(CollectiveTask, MiTask):
    name = "Path"

    def task(self, dir: DirectoryGroupDecision):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.D_file(dir).open("r") as f:
            D = from_csv(f)

        with self.Mc_file(dir).open("r") as f:
            Mc = cast(SRMPModel, model_from_json(f.read()))

        start_time = time.process_time()
        path = compute_preference_path(Mc, D, A)
        computation_time = time.process_time() - start_time

        for t, preferences in enumerate(path):
            with self.P_file(dir, t).open("w") as f:
                to_csv(preferences, f)

        dir.csv_files["path"].queue.put(
            {
                PathFieldnames.M: self.m,
                PathFieldnames.N_tr: self.ntr,
                PathFieldnames.Atr_id: self.Atr_id,
                PathFieldnames.Ko: self.ko,
                PathFieldnames.Mo_id: self.Mo_id,
                PathFieldnames.Group_size: self.group_size,
                PathFieldnames.Gen: self.gen,
                PathFieldnames.Mi_id: self.Mi_id,
                PathFieldnames.N_bc: self.nbc,
                PathFieldnames.Same_alt: self.same_alt,
                PathFieldnames.D_id: self.D_id,
                PathFieldnames.Config: self.config,
                PathFieldnames.It: self.it,
                PathFieldnames.Dm_id: self.dm_id,
                PathFieldnames.Time: computation_time,
                PathFieldnames.Length: t,
            }
        )

    def P_file(self, dir: DirectoryGroupDecision, t: int):
        return dir.P(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.gen,
            self.dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.it,
            t,
        )

    def D_file(self, dir: DirectoryGroupDecision):
        return super().D_file(dir, self.dm_id)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.P_file(dir, 0).exists()


@dataclass(frozen=True)
class AcceptTask(PreferencePathTask):
    name = "Accept"
    accept: AcceptHyperparameters
    t: int = field(hash=False)

    def task(self, dir: DirectoryGroupDecision):
        with self.Mi_file(dir, self.dm_id).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.P_file(dir).open("r") as f:
            D = from_csv(f)

        best_model, best_fitness, time = learn_mip(
            GroupModelEnum.SRMP,
            self.ko,
            A,
            [D],
            rng(0),
            0,
            self.config.max_time,
            reference_model=Mi,
            gamma=self.config.gamma,
            profiles_amp=self.accept.P,
            weights_amp=self.accept.W,
            lexicographic_order_distance=self.accept.L,
        )

        return best_model is not None

    def P_file(self, dir: DirectoryGroupDecision):
        return super().P_file(dir, self.t)

    def done(self, *args, **kwargs):
        return False
