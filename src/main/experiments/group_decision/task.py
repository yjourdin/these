import csv
from dataclasses import dataclass, field, replace
from typing import cast

from mcda.relations import PreferenceStructure
from pandas import read_csv

from ....mip.main import learn_mip
from ....models import GroupModelEnum, model_from_json
from ....performance_table.normal_performance_table import NormalPerformanceTable
from ....preference_path.main import compute_model_path, compute_preference_path
from ....preference_structure.generate import random_comparisons
from ....preference_structure.io import from_csv, to_csv
from ....random import Seed, rng
from ....random import seed as random_seed
from ....srmp.model import SRMPModel
from ...task import SeedTask
from ..elicitation.config import MIPConfig
from .directory import DirectoryGroupDecision
from .fieldnames import (
    AcceptFieldnames,
    CleanFieldnames,
    CollectiveFieldnames,
    PathFieldnames,
)
from .fields import GroupParameters


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
    group: GroupParameters
    Mi_id: int = field(hash=False)

    def Mi_file(self, dir: DirectoryGroupDecision, dm_id: int):
        return dir.Mi(
            self.m,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
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
            Mo, self.group.gen.P, self.group.gen.W, self.group.gen.L, rng=self.rng(seed)
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
            self.group,
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

        D = random_comparisons(A, Mi, self.nbc, rng=rng)

        with self.D_file(dir, self.dm_id, 0).open("w") as f:
            to_csv(D, f)

        return D

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.D_file(dir, self.dm_id, 0).exists()


@dataclass(frozen=True)
class CollectiveTask(AbstractDTask):
    name = "Collective"
    config: MIPConfig
    Mc_id: int = field(hash=False)
    path: bool = field(hash=False)
    P_id: int = field(hash=False)
    it: int  # = field(hash=False)

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, max_time: int | None = None
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

        R: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            for it in range(1, self.it + 1):
                # if (RP_file := self.RP_file(dir, dm_id, it)).exists():
                #     with RP_file.open("r") as f:
                #         R.append(from_csv(f))
                if (RC_file := self.RC_file(dir, dm_id, it)).exists():
                    with RC_file.open("r") as f:
                        if (RC := from_csv(f)) not in R:
                            R.append(RC)

        rng_lex, rng_mip = self.rng(seed).spawn(2)

        best_model, best_fitness, time = learn_mip(
            GroupModelEnum.SRMP,
            self.ko,
            A,
            D,
            rng_lex,
            random_seed(rng_mip),
            min(max_time, self.config.max_time)
            if max_time is not None
            else self.config.max_time,
            True,
            C,
            R,
            gamma=self.config.gamma,
        )

        if best_model is not None:
            with self.Mc_file(dir).open("w") as f:
                f.write(best_model.to_json())

        dir.csv_files["collective"].queue.put(
            {
                CollectiveFieldnames.M: self.m,
                CollectiveFieldnames.N_tr: self.ntr,
                CollectiveFieldnames.Atr_id: self.Atr_id,
                CollectiveFieldnames.Ko: self.ko,
                CollectiveFieldnames.Mo_id: self.Mo_id,
                CollectiveFieldnames.Group_size: self.group_size,
                CollectiveFieldnames.Group: self.group,
                CollectiveFieldnames.Mi_id: self.Mi_id,
                CollectiveFieldnames.N_bc: self.nbc,
                CollectiveFieldnames.Same_alt: self.same_alt,
                CollectiveFieldnames.D_id: self.D_id,
                CollectiveFieldnames.Config: self.config,
                CollectiveFieldnames.Mc_id: self.Mc_id,
                CollectiveFieldnames.Path: self.path,
                CollectiveFieldnames.P_id: self.P_id,
                CollectiveFieldnames.It: self.it,
                CollectiveFieldnames.Time: time,
                CollectiveFieldnames.Fitness: best_fitness,
            }
        )

        return best_model is not None

    def C_file(self, dir: DirectoryGroupDecision):
        return dir.C(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.path,
            self.P_id,
            self.it,
        )

    def RP_file(self, dir: DirectoryGroupDecision, dm_id: int, it: int):
        return dir.RP(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.path,
            self.P_id,
            it,
        )

    def RC_file(self, dir: DirectoryGroupDecision, dm_id: int, it: int):
        return dir.RC(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.path,
            self.P_id,
            it,
        )

    def Mc_file(self, dir: DirectoryGroupDecision):
        return dir.Mc(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            self.path,
            self.P_id,
            self.it,
        )

    def D_file(self, dir: DirectoryGroupDecision, dm_id: Seed):
        return super().D_file(dir, dm_id, self.it)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.Mc_file(dir).exists()


@dataclass(frozen=True)
class PreferencePathTask(CollectiveTask, MiTask):
    name = "Path"

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, max_time: int | None = None
    ):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.D_file(dir).open("r") as f:
            D = from_csv(f)

        with self.Mc_file(dir).open("r") as f:
            Mc = SRMPModel.from_json(f.read())

        R = []
        for RP_file in self.RP_file(dir).parent.iterdir():
            with RP_file.open("r") as f:
                R.append(from_csv(f))

        model_path, time = compute_model_path(
            Mc,
            D,
            A,
            self.rng(seed),
            min(max_time, self.config.max_time)
            if max_time is not None
            else self.config.max_time,
        )
        preference_path = compute_preference_path(model_path, D, A, R)

        t = None
        for t, preferences in enumerate(preference_path):
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
                PathFieldnames.Group: self.group,
                PathFieldnames.Mi_id: self.Mi_id,
                PathFieldnames.N_bc: self.nbc,
                PathFieldnames.Same_alt: self.same_alt,
                PathFieldnames.D_id: self.D_id,
                PathFieldnames.Config: self.config,
                PathFieldnames.Mc_id: self.Mc_id,
                PathFieldnames.P_id: self.P_id,
                PathFieldnames.It: self.it,
                PathFieldnames.Dm_id: self.dm_id,
                PathFieldnames.Time: time,
                PathFieldnames.Length: t,
                PathFieldnames.Model_Length: len(model_path),
            }
        )

    def P_file(self, dir: DirectoryGroupDecision, t: int, path: bool = True):
        return dir.P(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            self.dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            path,
            self.P_id,
            self.it,
            t,
        )

    def D_file(self, dir: DirectoryGroupDecision):
        return super().D_file(dir, self.dm_id)

    def RP_file(self, dir: DirectoryGroupDecision):
        return super().RP_file(dir, self.dm_id, self.it)

    def done(self, dir: DirectoryGroupDecision, *args, **kwargs):
        return self.P_file(dir, 0).exists()


@dataclass(frozen=True)
class NoPathTask(CollectiveTask, MiTask):
    name = "NoPath"

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, max_time: int | None = None
    ):
        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.Mc_file(dir).open("r") as f:
            Mc = SRMPModel.from_json(f.read())

        with self.D_file(dir).open("r") as f:
            D = from_csv(f)

        Dc = random_comparisons(A, Mc, len(D), [r.elements for r in D])

        with self.P_file(dir, 0).open("w") as f:
            to_csv(D, f)
        
        with self.P_file(dir, 1).open("w") as f:
            to_csv(Dc, f)

    def P_file(self, dir: DirectoryGroupDecision, t: int):
        return dir.P(
            self.m,
            self.ntr,
            self.Atr_id,
            self.ko,
            self.Mo_id,
            self.group_size,
            self.group,
            self.dm_id,
            self.Mi_id,
            self.nbc,
            self.same_alt,
            self.D_id,
            self.config,
            self.Mc_id,
            False,
            self.P_id,
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
            profiles_amp=self.group.accept.P,
            weights_amp=self.group.accept.W,
            lexicographic_order_distance=self.group.accept.L,
        )

        dir.csv_files["accept"].queue.put(
            {
                AcceptFieldnames.M: self.m,
                AcceptFieldnames.N_tr: self.ntr,
                AcceptFieldnames.Atr_id: self.Atr_id,
                AcceptFieldnames.Ko: self.ko,
                AcceptFieldnames.Mo_id: self.Mo_id,
                AcceptFieldnames.Group_size: self.group_size,
                AcceptFieldnames.Group: self.group,
                AcceptFieldnames.Mi_id: self.Mi_id,
                AcceptFieldnames.N_bc: self.nbc,
                AcceptFieldnames.Same_alt: self.same_alt,
                AcceptFieldnames.D_id: self.D_id,
                AcceptFieldnames.Config: self.config,
                AcceptFieldnames.Mc_id: self.Mc_id,
                AcceptFieldnames.Path: self.path,
                AcceptFieldnames.P_id: self.P_id,
                AcceptFieldnames.It: self.it,
                AcceptFieldnames.Dm_id: self.dm_id,
                AcceptFieldnames.T: self.t,
                AcceptFieldnames.Accept: best_model is not None,
            }
        )

        return best_model is not None

    def P_file(self, dir: DirectoryGroupDecision):
        return super().P_file(dir, self.t, self.path)

    def done(self, *args, **kwargs):
        return False


@dataclass(frozen=True)
class CleanTask(PreferencePathTask):
    name = "Clean"

    def task(self, dir: DirectoryGroupDecision):
        with self.Mi_file(dir, self.dm_id).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        with self.A_train_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        count = 0
        total = 0
        for it in range(1, self.it + 1):
            if (RC_file := self.RC_file(dir, it)).exists():
                total += 1

                with RC_file.open("r") as f:
                    R = from_csv(f)

                best_model, best_fitness, time = learn_mip(
                    GroupModelEnum.SRMP,
                    self.ko,
                    A,
                    [R],
                    rng(0),
                    0,
                    self.config.max_time,
                    reference_model=Mi,
                    gamma=self.config.gamma,
                    profiles_amp=self.group.accept.P,
                    weights_amp=self.group.accept.W,
                    lexicographic_order_distance=self.group.accept.L,
                )

                if best_model is not None:
                    count += 1
                    RC_file.unlink()

        dir.csv_files["clean"].queue.put(
            {
                CleanFieldnames.M: self.m,
                CleanFieldnames.N_tr: self.ntr,
                CleanFieldnames.Atr_id: self.Atr_id,
                CleanFieldnames.Ko: self.ko,
                CleanFieldnames.Mo_id: self.Mo_id,
                CleanFieldnames.Group_size: self.group_size,
                CleanFieldnames.Group: self.group,
                CleanFieldnames.Mi_id: self.Mi_id,
                CleanFieldnames.N_bc: self.nbc,
                CleanFieldnames.Same_alt: self.same_alt,
                CleanFieldnames.D_id: self.D_id,
                CleanFieldnames.Config: self.config,
                CleanFieldnames.Mc_id: self.Mc_id,
                CleanFieldnames.Path: self.path,
                CleanFieldnames.P_id: self.P_id,
                CleanFieldnames.It: self.it,
                CleanFieldnames.Dm_id: self.dm_id,
                CleanFieldnames.Removed: count,
                CleanFieldnames.Total: total,
            }
        )

    def RC_file(self, dir: DirectoryGroupDecision, it: int):
        return super().RC_file(dir, self.dm_id, it)

    def done(self, *args, **kwargs):
        return False
