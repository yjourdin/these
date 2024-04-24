from typing import Literal

from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

from mip.main import learn_mip
from model import ModelType
from performance_table.generate import random_alternatives
from performance_table.normal_performance_table import NormalPerformanceTable
from preference_structure.fitness import fitness_outranking
from preference_structure.generate import noisy_comparisons, random_comparisons
from preference_structure.io import from_csv, to_csv
from rmp.generate import random_rmp
from rmp.model import RMPModel
from run.config import SAConfig
from sa.main import learn_sa
from srmp.generate import random_srmp
from srmp.model import SRMPModel

from .path import Directory


def create_A_train(m: int, n: int, id: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_train_file(m, n, id).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_A_test(m: int, n: int, id: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_test_file(m, n, id).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_Mo(
    m: int, model: ModelType, k: int, id: int, dir: Directory, rng: Generator
):
    match model:
        case "RMP":
            Mo = random_rmp(k, m, rng)
        case "SRMP":
            Mo = random_srmp(k, m, rng)
    with dir.Mo_file(m, model, k, id).open("w") as f:
        f.write(Mo.to_json())


def create_D(
    m: int,
    n_tr: int,
    Atr_id: int,
    Mo: ModelType,
    ko: int,
    Mo_id: int,
    n: int,
    error: float,
    dir: Directory,
    rng: Generator,
):
    with dir.Mo_file(m, Mo, ko, Mo_id).open("r") as f:
        match Mo:
            case "RMP":
                model = RMPModel.from_json(f.read())
            case "SRMP":
                model = SRMPModel.from_json(f.read())

    with dir.A_train_file(m, n_tr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D_file(m, n_tr, Atr_id, Mo, ko, Mo_id, n, error).open("w") as f:
        f.write(to_csv(D))


def run_SA(
    m: int,
    n_tr: int,
    Atr_id: int,
    Mo: ModelType,
    ko: int,
    Mo_id: int,
    n: int,
    e: float,
    Me: ModelType,
    ke: int,
    config_id: int,
    config: SAConfig,
    dir: Directory,
    rng: Generator,
):
    with dir.A_train_file(m, n_tr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(m, n_tr, Atr_id, Mo, ko, Mo_id, n, e).open("r") as f:
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

    with dir.Me_file(
        m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, Me, ke, "SA", config_id
    ).open("w") as f:
        f.write(best_model.to_json())

    return (time, it, best_fitness)


def run_MIP(
    m: int,
    n_tr: int,
    Atr_id: int,
    Mo: ModelType,
    ko: int,
    Mo_id: int,
    n: int,
    e: float,
    ke: int,
    dir: Directory,
    seed: int,
):
    with dir.A_train_file(m, n_tr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(m, n_tr, Atr_id, Mo, ko, Mo_id, n, e).open("r") as f:
        D = from_csv(f.read())

    best_model, best_fitness, time = learn_mip(ke, A, D, seed=seed)

    with dir.Me_file(m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, "SRMP", ke, "MIP", 0).open(
        "w"
    ) as f:
        f.write(best_model.to_json())

    return (time, best_fitness)


def run_test(
    m: int,
    n_tr: int,
    Atr_id: int,
    Mo_type: ModelType,
    ko: int,
    Mo_id: int,
    n: int,
    e: float,
    Me_type: ModelType,
    ke: int,
    method: Literal["MIP", "SA"],
    config: int,
    n_te: int,
    Ate_id: int,
    dir: Directory,
):
    with dir.A_test_file(m, n_te, Ate_id).open("r") as f:
        A_test = NormalPerformanceTable(read_csv(f, header=None))

    with dir.Mo_file(m, Mo_type, ko, Mo_id).open("r") as f:
        match Mo_type:
            case "RMP":
                Mo = RMPModel.from_json(f.read())
            case "SRMP":
                Mo = SRMPModel.from_json(f.read())

    with dir.Me_file(
        m, n_tr, Atr_id, Mo_type, ko, Mo_id, n, e, Me_type, ke, method, config
    ).open("r") as f:
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
