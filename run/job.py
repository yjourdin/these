from typing import Literal

from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

from preference_structure.fitness import fitness_outranking
from mip.main import learn_mip
from model import ModelType
from performance_table.generate import random_alternatives
from performance_table.normal_performance_table import NormalPerformanceTable
from preference_structure.generate import noisy_comparisons, random_comparisons
from preference_structure.io import from_csv, to_csv
from rmp.generate import random_rmp
from rmp.model import RMPModel
from run.config import SAConfig
from sa.main import learn_sa
from srmp.generate import random_srmp
from srmp.model import SRMPModel

from .path import Directory


def create_A_train(i: int, m: int, n: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_train_file(i, m, n).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_A_test(i: int, m: int, n: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_test_file(i, m, n).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_Mo(i: int, m: int, model: ModelType, k: int, dir: Directory, rng: Generator):
    match model:
        case "RMP":
            Mo = random_rmp(k, m, rng)
        case "SRMP":
            Mo = random_srmp(k, m, rng)
    with dir.Mo_file(i, m, model, k).open("w") as f:
        f.write(Mo.to_json())


def create_D(
    i: int,
    m: int,
    n_tr: int,
    Mo: ModelType,
    ko: int,
    n: int,
    error: float,
    dir: Directory,
    rng: Generator,
):
    with dir.Mo_file(i, m, Mo, ko).open("r") as f:
        match Mo:
            case "RMP":
                model = RMPModel.from_json(f.read())
            case "SRMP":
                model = SRMPModel.from_json(f.read())

    with dir.A_train_file(i, m, n_tr).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D_file(i, m, n_tr, Mo, ko, n, error).open("w") as f:
        f.write(to_csv(D))


def run_SA(
    i: int,
    m: int,
    n_tr: int,
    Mo: ModelType,
    ko: int,
    n: int,
    e: float,
    Me: ModelType,
    ke: int,
    config_id: int,
    config: SAConfig,
    dir: Directory,
    rng: Generator,
):
    with dir.A_train_file(i, n_tr, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(i, n_tr, m, Mo, ko, n, e).open("r") as f:
        D = from_csv(f.read())

    rng_init, rng_sa = rng.spawn(2)
    best_model, best_fitness, time, it = learn_sa(
        Me,
        ke,
        A,
        D,
        config.T0_coef / n,
        config.alpha,
        config.amp,
        rng_init,
        rng_sa,
        max_iter=config.max_iter,
    )

    with dir.Me_file(i, n_tr, m, Mo, ko, n, e, Me, ke, "SA", config_id).open("w") as f:
        f.write(best_model.to_json())

    return (time, it, best_fitness)


def run_MIP(
    i: int,
    m: int,
    n_tr: int,
    Mo: ModelType,
    ko: int,
    n: int,
    e: float,
    ke: int,
    dir: Directory,
    seed: int,
):
    with dir.A_train_file(i, n_tr, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(i, n_tr, m, Mo, ko, n, e).open("r") as f:
        D = from_csv(f.read())

    best_model, best_fitness, time = learn_mip(ke, A, D, seed=seed)

    with dir.Me_file(i, n_tr, m, Mo, ko, n, e, "SRMP", ke, "MIP").open("w") as f:
        f.write(best_model.to_json())

    return (time, best_fitness)


def run_test(
    i: int,
    m: int,
    n_tr: int,
    Mo_type: ModelType,
    ko: int,
    n: int,
    e: float,
    Me_type: ModelType,
    ke: int,
    method: Literal["MIP", "SA"],
    config: int,
    n_te: int,
    dir: Directory,
):
    with dir.A_test_file(i, n_te, m).open("r") as f:
        A_test = NormalPerformanceTable(read_csv(f, header=None))

    with dir.Mo_file(i, m, Mo_type, ko).open("r") as f:
        match Mo_type:
            case "RMP":
                Mo = RMPModel.from_json(f.read())
            case "SRMP":
                Mo = SRMPModel.from_json(f.read())

    with dir.Me_file(i, n_tr, m, Mo_type, ko, n, e, Me_type, ke, method, config).open(
        "r"
    ) as f:
        match Me_type:
            case "RMP":
                Me = RMPModel.from_json(f.read())
            case "SRMP":
                Me = SRMPModel.from_json(f.read())

    Ro = Mo.rank(A_test)
    Re = Me.rank(A_test)

    test_fitness = fitness_outranking(Ro, Re)

    kendall_tau = kendalltau(Ro.data, Re.data).statistic

    return (test_fitness, kendall_tau)
