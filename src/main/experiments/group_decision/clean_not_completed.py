from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.main.experiments.elicitation.config import SAConfig
from src.main.experiments.group_decision.directory import DirectoryGroupDecision
from src.main.experiments.group_decision.fields import (
    GroupParameters,
    ParametersDeviation,
)
from src.main.experiments.group_decision.task import PreferencePathTask
from src.methods import MethodEnum
from src.models import ModelEnum

DIR = DirectoryGroupDecision(Path().name, Path().parent)

df = pd.read_csv(Path("compromise_results.csv"))


class Exp(NamedTuple):
    M: int
    Ko: int
    Group_size: int
    N_bc: int
    Nb_Mcp: int


# Exp = namedtuple("Exp", )
exps: set[Exp] = set()

for m in range(3, 7):
    for k in range(1, 4):
        for g in range(2, 8, 2):
            for n in range(100, 600, 100):
                for p in range(1, 4):
                    exps.add(Exp(m, k, g, n, p))

for exp in df.itertuples(False):
    exps.remove(Exp(exp.M, exp.Ko, exp.Group_size, exp.N_bc, exp.Nb_Mcp))

for exp in exps:
    csvs = [
        Path("collective_results.csv"),
        Path("path_results.csv"),
        Path("distance.csv"),
        Path("changes_results.csv"),
    ]
    for csv in csvs:
        dff = pd.read_csv(csv)
        if not (
            d := dff[
                (dff["M"] == exp.M)
                & (dff["Ko"] == exp.Ko)
                & (dff["Group_size"] == exp.Group_size)
                & (dff["N_bc"] == exp.N_bc)
                & (dff["Nb_Mcp"] == exp.Nb_Mcp)
            ]
        ).empty:
            dff = dff.drop(d.index)
            dff.to_csv(csv, index=False)

    for dm in range(exp.Group_size):
        print(exp, dm)
        it = 0
        task = PreferencePathTask(
            exp.M,
            1000,
            0,
            ModelEnum.RMP,
            exp.Ko,
            False,
            0,
            exp.Group_size,
            GroupParameters(0.1, ParametersDeviation(0.1, 0.1, 0.1, 0)),
            0,
            dm,
            exp.N_bc,
            True,
            0,
            0,
            MethodEnum.SA,
            SAConfig(),
            exp.Nb_Mcp,
            0,
            True,
            0,
            it,
        )
        print(task.Di_file(DIR, dm).resolve())
        while task.Di_file(DIR, dm).exists():
            print(exp, dm, it)
            task.Di_file(DIR, dm).unlink(True)
            task.C_file(DIR).unlink(True)
            task.Cr_file(DIR).unlink(True)
            task.Da_file(DIR).unlink(True)
            task.Dr_file(DIR, it).unlink(True)
            for id in range(exp.Nb_Mcp):
                task.Mcp_file(DIR, id).unlink(True)
                task.Dcp_file(DIR, id).unlink(True)
            task.Mc_file(DIR).unlink(True)
            task.Dc_file(DIR).unlink(True)
            t = 0
            while task.P_file(DIR, t).exists():
                print(exp, dm, it, t)
                task.P_file(DIR, t).unlink(True)
                task.Mp_file(DIR, t).unlink(True)
                task.Dp_file(DIR, t).unlink(True)
                t += 1
            task = replace(task, it=it + 1)
