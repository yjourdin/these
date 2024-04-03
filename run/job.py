from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

from fitness import fitness_ranking
from mip.main import learn_mip
from model import ModelType
from performance_table.generate import random_alternatives
from performance_table.normal_performance_table import NormalPerformanceTable
from preference_structure.generate import noisy_comparisons, random_comparisons
from preference_structure.io import from_csv, to_csv
from rmp.generate import random_rmp
from rmp.model import RMPModel
from sa.main import learn_sa
from srmp.generate import random_srmp
from srmp.model import SRMPModel

from .path import Directory


def create_A_train(n: int, m: int, i: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_train_file(i, n, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_A_test(n: int, m: int, i: int, dir: Directory, rng: Generator):
    A = random_alternatives(n, m, rng)
    with dir.A_test_file(i, n, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_Mo(model: ModelType, k: int, m: int, i: int, dir: Directory, rng: Generator):
    match model:
        case "RMP":
            Mo = random_rmp(k, m, rng)
        case "SRMP":
            Mo = random_srmp(k, m, rng)
    with dir.Mo_file(i, m, model, k).open("w") as f:
        f.write(Mo.to_json())


def create_D(
    n: int,
    error: float,
    Mo: ModelType,
    ko: int,
    m: int,
    n_tr: int,
    i: int,
    dir: Directory,
    rng: Generator,
):
    with dir.Mo_file(i, m, Mo, ko).open("r") as f:
        match Mo:
            case "RMP":
                model = RMPModel.from_json(f.read())
            case "SRMP":
                model = SRMPModel.from_json(f.read())

    with dir.A_train_file(i, n_tr, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D_file(i, n_tr, m, Mo, ko, n, error).open("w") as f:
        f.write(to_csv(D))


def run_SA(
    Me: ModelType,
    ke: int,
    n: int,
    e: float,
    Mo: ModelType,
    ko: int,
    m: int,
    n_tr: int,
    i: int,
    T0: float,
    Tf: float,
    alpha: float,
    amp: float,
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
        T0,
        alpha,
        amp,
        rng_init,
        rng_sa,
        Tf=Tf,
    )

    with dir.Me_file(i, n_tr, m, Mo, ko, n, e, Me, ke).open("w") as f:
        f.write(best_model.to_json())

    return (time, it, best_fitness)


def run_MIP(
    ke: int,
    n: int,
    e: float,
    Mo: ModelType,
    ko: int,
    m: int,
    n_tr: int,
    i: int,
    dir: Directory,
):
    with dir.A_train_file(i, n_tr, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(i, n_tr, m, Mo, ko, n, e).open("r") as f:
        D = from_csv(f.read())

    best_model, best_fitness, time = learn_mip(ke, A, D)

    with dir.Me_file(i, n_tr, m, Mo, ko, n, e, "SRMP", ke).open("w") as f:
        f.write(best_model.to_json())

    return (time, best_fitness)


def run_test(
    Me_type: ModelType,
    ke: int,
    n: int,
    e: float,
    Mo_type: ModelType,
    ko: int,
    m: int,
    n_te: int,
    n_tr: int,
    i: int,
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

    with dir.Me_file(i, n_tr, m, Mo_type, ko, n, e, Me_type, ke).open("r") as f:
        match Me_type:
            case "RMP":
                Me = RMPModel.from_json(f.read())
            case "SRMP":
                Me = SRMPModel.from_json(f.read())

    Ro = Mo.rank(A_test)
    Re = Me.rank(A_test)

    test_fitness = fitness_ranking(Ro, Re)

    kendall_tau = kendalltau(Ro, Re).statistic

    return (test_fitness, kendall_tau)
