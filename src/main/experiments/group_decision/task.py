import csv
from dataclasses import dataclass, field, replace
from typing import Any

from mcda.internal.core.relations import Relation
from mcda.relations import PreferenceStructure
from pandas import read_csv

from ....mip.main import learn_mip
from ....models import GroupModelEnum
from ....performance_table.normal_performance_table import NormalPerformanceTable
from ....preference_path.main import compute_model_path, compute_preference_path
from ....preference_structure.generate import random_comparisons
from ....preference_structure.io import from_csv, to_csv
from ....random import Seed, rng
from ....random import seed as random_seed
from ....srmp.model import SRMPModel
from ....utils import tolist
from ...task import SeedTask
from ..elicitation.config import MIPConfig
from .directory import DirectoryGroupDecision
from .fields import GroupParameters


@dataclass(frozen=True)
class AbstractMTask(SeedTask):
    m: int


@dataclass(frozen=True)
class ATask(AbstractMTask):
    name = "A"
    ntr: int
    Atr_id: int = field(hash=False)

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, *args: Any, **kwargs: Any
    ) -> Any:
        A = NormalPerformanceTable.random(self.ntr, self.m, self.rng(seed))

        with self.A_file(dir).open("w") as f:
            A.data.to_csv(f, header=False, index=False)

    def A_file(self, dir: DirectoryGroupDecision):
        return dir.A(self.m, self.ntr, self.Atr_id)

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
        return self.A_file(dir).exists()


@dataclass(frozen=True)
class MoTask(AbstractMTask):
    name = "Mo"
    ko: int
    fixed_lex_order: bool = field(hash=False)
    Mo_id: int = field(hash=False)

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, *args: Any, **kwargs: Any
    ) -> Any:
        Mo = SRMPModel.random(nb_profiles=self.ko, nb_crit=self.m, rng=self.rng(seed))

        if self.fixed_lex_order:
            Mo.lexicographic_order = self.lexicographic_order

        with self.Mo_file(dir).open("w") as f:
            f.write(Mo.to_json())

    @property
    def lexicographic_order(self) -> list[int]:
        return tolist(
            MoTask(self.m, self.ko, True, self.Mo_id)
            .rng(self.Mo_id)
            .permutation(self.ko)
        )

    def Mo_file(self, dir: DirectoryGroupDecision):
        return dir.Mo(self.m, self.ko, self.Mo_id)

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
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

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, *args: Any, **kwargs: Any
    ) -> Any:
        with self.Mo_file(dir).open("r") as f:
            Mo = SRMPModel.from_json(f.read())

        Mi = SRMPModel.from_reference(
            Mo,
            self.group.gen.P,
            self.group.gen.W,
            self.group.gen.L if not self.fixed_lex_order else 0,
            rng=self.rng(seed),
        )

        with self.Mi_file(dir, self.dm_id).open("w") as f:
            f.write(Mi.to_json())

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
        return self.Mi_file(dir, self.dm_id).exists()


@dataclass(frozen=True)
class AbstractDTask(AbstractMiTask, ATask):
    nbc: int
    same_alt: bool
    D_id: int = field(hash=False)

    def D_file(self, dir: DirectoryGroupDecision, dm_id: int):
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
        )


@dataclass(frozen=True)
class DTask(AbstractDTask, MiTask):
    name = "D"

    def task(
        self, dir: DirectoryGroupDecision, seed: Seed, *args: Any, **kwargs: Any
    ) -> Any:
        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.Mi_file(dir, self.dm_id).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        if self.same_alt:
            rng = replace(self, dm_id=0).rng(seed)
        else:
            rng = self.rng(seed)

        D = random_comparisons(A, Mi, self.nbc, rng=rng)

        with self.D_file(dir).open("w") as f:
            to_csv(D, f)

        return D

    def D_file(self, dir: DirectoryGroupDecision, dm_id: int = int()):
        return super().D_file(dir, self.dm_id)

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
        return self.D_file(dir).exists()


@dataclass(frozen=True)
class MIPTask(AbstractDTask):
    name = "MIP"
    config: MIPConfig
    Mc_id: int = field(hash=False)
    path: bool = field(hash=False)
    P_id: int = field(hash=False)
    it: int  # = field(hash=False)

    def task(
        self,
        dir: DirectoryGroupDecision,
        seed: Seed,
        max_time: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        D: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            with self.Di_file(dir, dm_id).open("r") as f:
                D.append(from_csv(f))

        Acc_set: set[Relation] = set.intersection(*[set(d) for d in D])
        ACC = PreferenceStructure(list(Acc_set), validate=False)

        for d in D:
            d -= ACC

        C: list[int] = []
        with self.C_file(dir).open("r", newline="") as f:
            C_reader = csv.reader(f, dialect="unix")
            for changes in C_reader:
                C.append(int(changes[0]))

        R: list[PreferenceStructure] = []
        for dm_id in range(self.group_size):
            for it in range(self.it):
                if (Cr_file := self.Cr_file(dir, dm_id, it)).exists():
                    with Cr_file.open("r") as f:
                        if (RC := from_csv(f)) not in R:
                            R.append(RC)
                if (Dr_file := self.Dr_file(dir, dm_id, it)).exists():
                    with Dr_file.open("r") as f:
                        R.append(from_csv(f))

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
            self.lexicographic_order if self.fixed_lex_order else None,
            True,
            C,
            R,
            ACC,
            gamma=self.config.gamma,
        )

        if best_model is not None:
            with self.Mc_file(dir).open("w") as f:
                f.write(best_model.to_json())

            with self.Dc_file(dir).open("w") as f:
                to_csv(
                    random_comparisons(
                        A, best_model, pairs=D[0].elements_pairs_relations
                    ),
                    f,
                )

        csv_file = dir.csv_files["mip"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Ko=self.ko,
                Mo_id=self.Mo_id,
                Group_size=self.group_size,
                Group=self.group,
                Mi_id=self.Mi_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                D_id=self.D_id,
                Config=self.config,
                Mc_id=self.Mc_id,
                Path=self.path,
                P_id=self.P_id,
                It=self.it,
                Time=time,
                Fitness=best_fitness,
            )
        )

        return best_model is not None

    def Di_file(self, dir: DirectoryGroupDecision, dm_id: Seed):
        return dir.Di(
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
            self.it,
        )

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

    def Dr_file(self, dir: DirectoryGroupDecision, dm_id: int, it: int):
        return dir.Dr(
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

    def Cr_file(self, dir: DirectoryGroupDecision, dm_id: int, it: int):
        return dir.Cr(
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

    def Dc_file(self, dir: DirectoryGroupDecision):
        return dir.Dc(
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

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
        return self.Mc_file(dir).exists()


@dataclass(frozen=True)
class AcceptMcTask(MIPTask, MiTask):
    name = "AcceptMc"

    def task(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any) -> Any:
        with self.Mi_file(dir).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.Dc_file(dir).open("r") as f:
            Dc = from_csv(f)

        best_model, _best_fitness, _time = learn_mip(
            GroupModelEnum.SRMP,
            self.ko,
            A,
            [Dc],
            rng(0),
            0,
            self.config.max_time,
            self.lexicographic_order,
            reference_model=Mi,
            gamma=self.config.gamma,
            profiles_amp=self.group.accept.P,
            weights_amp=self.group.accept.W,
            lexicographic_order_distance=self.group.accept.L,
        )

        csv_file = dir.csv_files["accept"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Ko=self.ko,
                Mo_id=self.Mo_id,
                Group_size=self.group_size,
                Group=self.group,
                Mi_id=self.Mi_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                D_id=self.D_id,
                Config=self.config,
                Mc_id=self.Mc_id,
                Path=self.path,
                P_id=self.P_id,
                It=self.it,
                Dm_id=self.dm_id,
                T=0,
                Accept=best_model is not None,
            )
        )

        return best_model is not None

    def Mi_file(self, dir: DirectoryGroupDecision, dm_id: int = int()):
        return super().Mi_file(dir, self.dm_id)

    def Di_file(self, dir: DirectoryGroupDecision, dm_id: int = int()):
        return super().Di_file(dir, self.dm_id)

    def done(self, *args: Any, **kwargs: Any) -> bool:
        return False


@dataclass(frozen=True)
class PreferencePathTask(AcceptMcTask):
    name = "Path"

    def task(
        self,
        dir: DirectoryGroupDecision,
        seed: Seed,
        max_time: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.Di_file(dir).open("r") as f:
            D = from_csv(f)

        with self.Mc_file(dir).open("r") as f:
            Mc = SRMPModel.from_json(f.read())

        if self.path:
            R: list[PreferenceStructure] = []
            for Dr_file in self.Dr_file(dir).parent.iterdir():
                with Dr_file.open("r") as f:
                    R.append(from_csv(f))

            model_path, time = compute_model_path(
                Mc,
                D,
                A,
                self.rng(seed),
                min(max_time, self.config.max_time)
                if max_time is not None
                else self.config.max_time,
                self.fixed_lex_order,
            )
            preference_path = compute_preference_path(model_path, D, A, R)
        else:
            model_path = []
            time = 0
            preference_path = [
                D,
                random_comparisons(A, Mc, pairs=D.elements_pairs_relations),
            ]

        t = None
        for t, preferences in enumerate(preference_path):
            with self.P_file(dir, t).open("w") as f:
                to_csv(preferences, f)

        if self.path:
            csv_file = dir.csv_files["path"]
            csv_file.writerow(
                csv_file.fields(
                    M=self.m,
                    N_tr=self.ntr,
                    Atr_id=self.Atr_id,
                    Ko=self.ko,
                    Mo_id=self.Mo_id,
                    Group_size=self.group_size,
                    Group=self.group,
                    Mi_id=self.Mi_id,
                    N_bc=self.nbc,
                    Same_alt=self.same_alt,
                    D_id=self.D_id,
                    Config=self.config,
                    Mc_id=self.Mc_id,
                    Path=self.path,
                    P_id=self.P_id,
                    It=self.it,
                    Dm_id=self.dm_id,
                    Time=time,
                    Length=t,
                    Model_Length=len(model_path),
                )
            )

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
            self.path,
            self.P_id,
            self.it,
            t,
        )

    def Dr_file(self, dir: DirectoryGroupDecision, dm_id: int = int(), it: int = int()):
        return super().Dr_file(dir, self.dm_id, self.it)

    def done(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any):
        return self.P_file(dir, 0).exists()


@dataclass(frozen=True)
class AcceptPTask(PreferencePathTask):
    name = "AcceptP"
    t: int = field(hash=False)

    def task(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any) -> Any:
        with self.Mi_file(dir).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        with self.P_file(dir).open("r") as f:
            D = from_csv(f)

        best_model, _best_fitness, _time = learn_mip(
            GroupModelEnum.SRMP,
            self.ko,
            A,
            [D],
            rng(0),
            0,
            self.config.max_time,
            self.lexicographic_order,
            reference_model=Mi,
            gamma=self.config.gamma,
            profiles_amp=self.group.accept.P,
            weights_amp=self.group.accept.W,
            lexicographic_order_distance=self.group.accept.L,
        )

        csv_file = dir.csv_files["accept"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Ko=self.ko,
                Mo_id=self.Mo_id,
                Group_size=self.group_size,
                Group=self.group,
                Mi_id=self.Mi_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                D_id=self.D_id,
                Config=self.config,
                Mc_id=self.Mc_id,
                Path=self.path,
                P_id=self.P_id,
                It=self.it,
                Dm_id=self.dm_id,
                T=self.t,
                Accept=best_model is not None,
            )
        )

        return best_model is not None

    def P_file(self, dir: DirectoryGroupDecision, t: int = int()):
        return super().P_file(dir, self.t)

    def done(self, *args: Any, **kwargs: Any) -> bool:
        return False


@dataclass(frozen=True)
class CleanTask(PreferencePathTask):
    name = "Clean"

    def task(self, dir: DirectoryGroupDecision, *args: Any, **kwargs: Any) -> Any:
        with self.Mi_file(dir).open("r") as f:
            Mi = SRMPModel.from_json(f.read())

        with self.A_file(dir).open("r") as f:
            A = NormalPerformanceTable(read_csv(f, header=None))

        count = 0
        total = 0
        print("-----")
        for it in range(1, self.it + 1):
            if (Cr_file := self.Cr_file(dir, it)).exists():
                with Cr_file.open("r") as f:
                    R = from_csv(f)

                removed: list[Relation] = []
                for r in R:
                    print(r)
                    total += 1
                    best_model, _best_fitness, _time = learn_mip(
                        GroupModelEnum.SRMP,
                        self.ko,
                        A,
                        [PreferenceStructure(r)],
                        rng(0),
                        0,
                        self.config.max_time,
                        self.lexicographic_order,
                        reference_model=Mi,
                        gamma=self.config.gamma,
                        profiles_amp=self.group.accept.P,
                        weights_amp=self.group.accept.W,
                        lexicographic_order_distance=self.group.accept.L,
                    )

                    if best_model is not None:
                        removed.append(r)
                        count += 1

                if R_final := R - PreferenceStructure(removed, validate=False):
                    with Cr_file.open("w") as f:
                        to_csv(R_final, f)
                else:
                    Cr_file.unlink()

        csv_file = dir.csv_files["clean"]
        csv_file.writerow(
            csv_file.fields(
                M=self.m,
                N_tr=self.ntr,
                Atr_id=self.Atr_id,
                Ko=self.ko,
                Mo_id=self.Mo_id,
                Group_size=self.group_size,
                Group=self.group,
                Mi_id=self.Mi_id,
                N_bc=self.nbc,
                Same_alt=self.same_alt,
                D_id=self.D_id,
                Config=self.config,
                Mc_id=self.Mc_id,
                Path=self.path,
                P_id=self.P_id,
                It=self.it,
                Dm_id=self.dm_id,
                Removed=count,
                Total=total,
            )
        )

    def Cr_file(self, dir: DirectoryGroupDecision, it: int, *args: Any, **kwargs: Any):
        return super().Cr_file(dir, self.dm_id, it)

    def done(self, *args: Any, **kwargs: Any) -> bool:
        return False
