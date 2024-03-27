import logging
from multiprocessing import Queue

import numpy as np
from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

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
    logger = logging.getLogger("log")
    log_message = f"A_train (No: {i:2} M: {m:2})"
    logger.info(log_message + " running...")
    A = random_alternatives(n, m, rng)
    with dir.A_train_file(i, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)
    logger.info(log_message + " done")


def create_A_test(n: int, m: int, i: int, dir: Directory, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"A_test (No: {i:2} M: {m:2})"
    logger.info(log_message + " running...")
    A = random_alternatives(n, m, rng)
    with dir.A_test_file(i, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)
    logger.info(log_message + " done")


def create_Mo(model: ModelType, k: int, m: int, i: int, dir: Directory, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"Mo      (No: {i:2} M: {m:2} Mo: {model:4} Ko: {k:2})"
    logger.info(log_message + " running...")
    match model:
        case "RMP":
            Mo = random_rmp(k, m, rng)
        case "SRMP":
            Mo = random_srmp(k, m, rng)
    with dir.Mo_file(i, m, model, k).open("w") as f:
        f.write(Mo.to_json())
    logger.info(log_message + " done")


def create_D(
    n: int,
    error: float,
    Mo: ModelType,
    ko: int,
    m: int,
    i: int,
    dir: Directory,
    rng: Generator,
):
    logger = logging.getLogger("log")
    log_message = (
        f"D       (No: {i:2} M: {m:2} Mo: {Mo:4} Ko: {ko:2} N: {n:4} Error: {error:4})"
    )
    logger.info(log_message + " running...")
    with dir.Mo_file(i, m, Mo, ko).open("r") as f:
        match Mo:
            case "RMP":
                model = RMPModel.from_json(f.read())
            case "SRMP":
                model = SRMPModel.from_json(f.read())

    with dir.A_train_file(i, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D_file(i, m, Mo, ko, n, error).open("w") as f:
        f.write(to_csv(D))
    logger.info(log_message + " done")


def create_Me(
    Me: ModelType,
    ke: int,
    n: int,
    e: float,
    Mo: ModelType,
    ko: int,
    m: int,
    i: int,
    T0: float,
    Tf: float,
    alpha: float,
    dir: Directory,
    rng: Generator,
    results_queue: Queue,
):
    logger = logging.getLogger("log")
    log_message = (
        f"Me      ("
        f"No: {i:2} "
        f"M: {m:2} "
        f"Mo: {Mo:4} "
        f"Ko: {ko:2} "
        f"N: {n:4} "
        f"Error: {e:4} "
        f"Me: {Me:4} "
        f"Ke: {ke:2})"
    )
    logger.info(log_message + " running...")
    with dir.A_train_file(i, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    with dir.D_file(i, m, Mo, ko, n, e).open("r") as f:
        D = from_csv(f.read())

    rng_init, rng_sa = rng.spawn(2)
    sa = learn_sa(
        Me,
        ke,
        A,
        D,
        T0,
        alpha,
        rng_init,
        rng_sa,
        Tf=Tf,
    )

    with dir.Me_file(i, Mo, m, ko, n, e, Me, ke).open("w") as f:
        f.write(sa.best_sol.to_json())

    results_queue.put(
        f"{i},{m},{Mo},{ko},{n},{e},{Me},{ke},{sa.time},{sa.it},{1-sa.best_objective}\n"
    )
    logger.info(log_message + " done")


def compute_test(
    Me_type: ModelType,
    ke: int,
    n: int,
    e: float,
    Mo_type: ModelType,
    ko: int,
    m: int,
    i: int,
    dir: Directory,
    results_queue: Queue,
):
    logger = logging.getLogger("log")
    log_message = (
        f"Test    ("
        f"No: {i:2} "
        f"M: {m:2} "
        f"Mo: {Mo_type:4} "
        f"Ko: {ko:2} "
        f"N: {n:4} "
        f"Error: {e:4} "
        f"Me: {Me_type:4} "
        f"Ke: {ke:2})"
    )
    logger.info(log_message + " running...")
    with dir.A_test_file(i, m).open("r") as f:
        A_test = NormalPerformanceTable(read_csv(f, header=None))

    with dir.Mo_file(i, m, Mo_type, ko).open("r") as f:
        match Mo_type:
            case "RMP":
                Mo = RMPModel.from_json(f.read())
            case "SRMP":
                Mo = SRMPModel.from_json(f.read())

    with dir.Me_file(i, Mo_type, m, ko, n, e, Me_type, ke).open("r") as f:
        match Me_type:
            case "RMP":
                Me = RMPModel.from_json(f.read())
            case "SRMP":
                Me = SRMPModel.from_json(f.read())

    Ro = Mo.rank(A_test).data.to_numpy()
    Re = Me.rank(A_test).data.to_numpy()

    outranking_o = np.less.outer(Ro, Ro).astype("int64", copy=False)
    outranking_e = np.less.outer(Re, Re).astype("int64", copy=False)

    outranking_o = outranking_o - outranking_o.transpose()
    outranking_e = outranking_e - outranking_e.transpose()

    ind = np.triu_indices(len(Ro), 1)

    test_fitness = np.equal(outranking_o[ind], outranking_e[ind]).sum() / len(ind[0])

    kendall_tau = kendalltau(Ro, Re).statistic

    results_queue.put(
        f"{i},{m},{Mo_type},{ko},{n},{e},{Me_type},{ke},{test_fitness},{kendall_tau}\n"
    )
    logger.info(log_message + " done")
