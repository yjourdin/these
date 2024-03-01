import argparse
import logging
import logging.config
import logging.handlers
import threading
from collections import defaultdict
from json import load
from multiprocessing import JoinableQueue, Manager, Process, Queue
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from pandas import read_csv
from scipy.stats import kendalltau

from performance_table.core import NormalPerformanceTable
from performance_table.generate import random_alternatives
from preference_structure.core import from_csv, to_csv
from preference_structure.generate import noisy_comparisons, random_comparisons
from rmp.generate import random_rmp
from rmp.model import RMPModel
from sa.main import learn_sa
from srmp.generate import random_srmp
from srmp.model import SRMPModel


def key2int(dct):
    """Transform dict keys to int if possible

    :param dct: Dict to modify
    :return : Modified dict
    """
    try:
        return {int(k): v for k, v in dct.items()}
    except ValueError:
        return dct


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of the experiment")
parser.add_argument(
    "args",
    type=argparse.FileType("r"),
    help="Arguments file",
)
parser.add_argument(
    "-j", "--jobs", default=35, type=int, help="Maximum number of parallel jobs"
)
ARGS = parser.parse_args()
vars(ARGS).update(load(ARGS.args, object_hook=key2int))


# Initialize random seed
rng = default_rng(ARGS.seed)


# Create directories
dir = Path(f"results/{ARGS.name}/")
dir.mkdir()
(dir / "A_train").mkdir()
(dir / "A_test").mkdir()
(dir / "Mo").mkdir()
(dir / "Me").mkdir()
(dir / "D_train").mkdir()


# Path substitution
def path_A_train(i: int, m: int):
    return dir / "A_train" / f"No_{i}_M_{m}.csv"


def path_A_test(i: int, m: int):
    return dir / "A_test" / f"No_{i}_M_{m}.csv"


def path_Mo(i: int, m: int, k: int):
    return dir / "Mo" / f"No_{i}_M_{m}_K_{k}.json"


def path_D(i: int, m: int, k: int, n: int, e: float):
    return dir / "D_train" / f"No_{i}_M_{m}_Ko_{k}_N_{n}_E_{e}.csv"


def path_Me(i: int, m: int, ko: int, n: int, e: float, ke: int):
    return dir / "Me" / f"No_{i}_M_{m}_Ko_{ko}_N_{n}_E_{e}_Ke_{ke}.json"


# Jobs
def create_A_train(n: int, m: int, i: int, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"A_train (No: {i:2} M: {m:2})"
    logger.info(log_message + " running...")
    A = random_alternatives(n, m, rng)
    with path_A_train(i, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)
    logger.info(log_message + " done")


def create_A_test(n: int, m: int, i: int, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"A_test (No: {i:2} M: {m:2})"
    logger.info(log_message + " running...")
    A = random_alternatives(n, m, rng)
    with path_A_test(i, m).open("w") as f:
        A.data.to_csv(f, header=False, index=False)
    logger.info(log_message + " done")


def create_Mo(k: int, m: int, i: int, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"Mo      (No: {i:2} M: {m:2} Ko: {k:2})"
    logger.info(log_message + " running...")
    match ARGS.Mo:
        case "RMP":
            Mo = random_rmp(k, m, rng)
        case "SRMP":
            Mo = random_srmp(k, m, rng)
    with path_Mo(i, m, k).open("w") as f:
        f.write(Mo.to_json())
    logger.info(log_message + " done")


def create_D(n: int, error: float, k: int, m: int, i: int, rng: Generator):
    logger = logging.getLogger("log")
    log_message = f"D       (No: {i:2} M: {m:2} Ko: {k:2} N: {n:4} Error: {error:4})"
    logger.info(log_message + " running...")
    with path_Mo(i, m, k).open("r") as f:
        match ARGS.Mo:
            case "RMP":
                model = RMPModel.from_json(f.read())
            case "SRMP":
                model = SRMPModel.from_json(f.read())

    with path_A_train(i, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with path_D(i, m, k, n, error).open("w") as f:
        f.write(to_csv(D))
    logger.info(log_message + " done")


def create_Me(lock, ke: int, n: int, e: float, ko: int, m: int, i: int, rng: Generator):
    logger = logging.getLogger("log")
    log_message = (
        f"Me      (No: {i:2} M: {m:2} Ko: {ko:2} N: {n:4} Error: {e:4} Ke: {ke:2})"
    )
    logger.info(log_message + " running...")
    with path_A_train(i, m).open("r") as f:
        A = NormalPerformanceTable(read_csv(f))

    with path_D(i, m, ko, n, e).open("r") as f:
        D = from_csv(f.read())

    N_bc = len(D)

    rng_init, rng_sa = rng.spawn(2)
    sa = learn_sa(
        ARGS.Me,
        ke,
        A,
        D,
        ARGS.T0[N_bc],
        ARGS.alpha,
        rng_init,
        rng_sa,
        Tf=ARGS.Tf[N_bc],
    )

    with path_Me(i, m, ko, n, e, ke).open("w") as f:
        f.write(sa.best_model.to_json())

    with lock:
        with (dir / "train_results.csv").open("a") as f:
            f.write(
                f"{i},{m},{ko},{n},{e},{ke},{sa.time},{sa.it},{1-sa.best_objective}\n"
            )
    logger.info(log_message + " done")


def compute_test(lock, ke: int, n: int, e: float, ko: int, m: int, i: int):
    logger = logging.getLogger("log")
    log_message = (
        f"Test    (No: {i:2} M: {m:2} Ko: {ko:2} N: {n:4} Error: {e:4} Ke: {ke:2})"
    )
    logger.info(log_message + " running...")
    with path_A_test(i, m).open("r") as f:
        A_test = NormalPerformanceTable(read_csv(f))

    with path_Mo(i, m, ko).open("r") as f:
        match ARGS.Mo:
            case "RMP":
                Mo = RMPModel.from_json(f.read())
            case "SRMP":
                Mo = SRMPModel.from_json(f.read())

    with path_Me(i, m, ko, n, e, ke).open("r") as f:
        match ARGS.Me:
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

    with lock:
        with (dir / "test_results.csv").open("a") as f:
            f.write(f"{i},{m},{ko},{n},{e},{ke},{test_fitness},{kendall_tau}\n")
    logger.info(log_message + " done")


# Tasks
Task = tuple[Any, ...]


def task_A_train(i: int, m: int) -> Task:
    return ("A_train", i, m)


def task_A_test(i: int, m: int) -> Task:
    return ("A_test", i, m)


def task_Mo(i: int, m: int, k: int) -> Task:
    return ("Mo", i, m, k)


def task_D(i: int, m: int, k: int, n: int, e: float) -> Task:
    return ("D", i, m, k, n, e)


def task_Me(i: int, m: int, ko: int, n: int, e: float, ke: int) -> Task:
    return ("Me", i, m, ko, n, e, ke)


def task_test(i: int, m: int, ko: int, n: int, e: float, ke: int) -> Task:
    return ("Test", i, m, ko, n, e, ke)


def compute_task(task: Task):
    match task:
        case ("A_train", i, m):
            create_A_train(ARGS.N_tr, m, i, rngs[i])
        case ("A_test", i, m):
            create_A_test(ARGS.N_te, m, i, rngs[i])
        case ("Mo", i, m, k):
            create_Mo(k, m, i, rngs[i])
        case ("D", i, m, k, n, e):
            create_D(n, e, k, m, i, rngs[i])
        case ("Me", i, m, ko, n, e, ke):
            create_Me(lock_train, ke, n, e, ko, m, i, rngs[i])
        case ("Test", i, m, ko, n, e, ke):
            compute_test(lock_test, ke, n, e, ko, m, i)
        case _:
            raise ValueError("Unknown task")


def next_tasks(task, done_dict):
    done_dict[task] = True
    tasks = []
    # logger = logging.getLogger("log")
    for next_task in succeed[task]:
        # for t in precede[next_task]:
        #     logger.info(str(task) + " precede " + str(next_task) + " succeed " + str(t) + ": " + str(done_dict.get(t, False)))
        if all([done_dict.get(t, False) for t in precede[next_task]]):
            tasks.append(next_task)
    return tasks


# Logging
def logger_thread(q):
    for record in iter(q.get, "STOP"):
        logger = logging.getLogger(record.name)
        logger.handle(record)


d = {
    "version": 1,
    "formatters": {
        "detailed": {
            "class": "logging.Formatter",
            "format": "%(asctime)s %(levelname)-8s %(processName)-10s %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": dir / "log.log",
            "mode": "w",
            "formatter": "detailed",
        },
    },
    "loggers": {"log": {"handlers": ["file"]}},
    "root": {"level": "ERROR", "handlers": ["console"]},
}


# Worker
def worker(task_queue, put_dict, done_dict, logging_queue):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    for task in iter(task_queue.get, "STOP"):
        try:
            compute_task(task)
            for t in next_tasks(task, done_dict):
                # logger = logging.getLogger("log")
                # logger.info(str(task) + " put " + str(t))
                if not put_dict.get(t, False):
                    task_queue.put(t)
                    put_dict[t] = True
            task_queue.task_done()
        except Exception as e:
            logger = logging.getLogger("log")
            logger.error(e)
            logger.info("Kill")
            break
        # print(task_queue.qsize())
    task_queue.task_done()


# Main

succeed: defaultdict[Task, list[Task]] = defaultdict(list)
precede: defaultdict[Task, list[Task]] = defaultdict(list)

rngs = rng.spawn(ARGS.N_exp)

task_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()


with Manager() as manager:
    done_dict = manager.dict()
    put_dict = manager.dict()
    lock_train = manager.Lock()
    lock_test = manager.Lock()

    for i in range(ARGS.N_exp):
        for m in ARGS.M:
            t_A_train = task_A_train(i, m)
            t_A_test = task_A_test(i, m)
            task_queue.put(t_A_train)
            task_queue.put(t_A_test)
            for ko in ARGS.Ko:
                t_Mo = task_Mo(i, m, ko)
                task_queue.put(t_Mo)
                for n_bc in ARGS.N_bc:
                    for e in ARGS.error:
                        t_D = task_D(i, m, ko, n_bc, e)
                        precede[t_D] += [t_A_train, t_Mo]
                        succeed[t_A_train] += [t_D]
                        succeed[t_Mo] += [t_D]
                        for ke in ARGS.Ke:
                            t_Me = task_Me(i, m, ko, n_bc, e, ke)
                            t_test = task_test(i, m, ko, n_bc, e, ke)
                            precede[t_Me] += [t_D]
                            succeed[t_D] += [t_Me]
                            precede[t_test] += [t_A_test, t_Me]
                            succeed[t_Me] += [t_test]
                            succeed[t_A_test] += [t_test]

    for i in range(ARGS.jobs):
        Process(
            target=worker, args=(task_queue, put_dict, done_dict, logging_queue)
        ).start()

    logging.config.dictConfig(d)
    logging_thread = threading.Thread(target=logger_thread, args=(logging_queue,))
    logging_thread.start()

    task_queue.join()
    for i in range(ARGS.jobs):
        task_queue.put("STOP")

    logging_queue.put("STOP")
    logging_thread.join()


# with Pool(ARGS.jobs) as pool:
#     rngs = rng.spawn(ARGS.N_exp)

#     res_A_train = pool.starmap_async(
#         create_A_train,
#         [(ARGS.N_tr, m, i, rngs[i]) for m in ARGS.M for i in range(ARGS.N_exp)],
#     )
#     res_A_test = pool.starmap_async(
#         create_A_test,
#         [(ARGS.N_te, m, i, rngs[i]) for m in ARGS.M for i in range(ARGS.N_exp)],
#     )
#     res_Mo = pool.starmap_async(
#         create_Mo,
#         [
#             (k, m, i, rngs[i])
#             for k in ARGS.Ko
#             for m in ARGS.M
#             for i in range(ARGS.N_exp)
#         ],
#     )
#     res_A_train.wait()
#     res_A_test.wait()
#     res_Mo.wait()

#     pool.starmap(
#         create_D,
#         [
#             (n, e, k, m, i, rngs[i])
#             for n in ARGS.N_bc
#             for e in ARGS.error
#             for k in ARGS.Ko
#             for m in ARGS.M
#             for i in range(ARGS.N_exp)
#         ],
#     )

#     with Manager() as manager:
#         lock = manager.Lock()

#         with (dir / "train_results.csv").open("w") as f:
#             f.write("No,M,Ko,N_bc,Error,Ke,Time,It,Fitness\n")
#         pool.starmap(
#             create_Me,
#             [
#                 (lock, ke, n, e, ko, m, i, rngs[i])
#                 for ke in ARGS.Ke
#                 for n in ARGS.N_bc
#                 for e in ARGS.error
#                 for ko in ARGS.Ko
#                 for m in ARGS.M
#                 for i in range(ARGS.N_exp)
#             ],
#         )

#         with (dir / "test_results.csv").open("w") as f:
#             f.write("No,M,Ko,N_bc,Error,Ke,Fitness,Kendall_tau\n")
#         pool.starmap(
#             compute_test,
#             [
#                 (lock, ke, n, e, ko, m, i)
#                 for ke in ARGS.Ke
#                 for n in ARGS.N_bc
#                 for e in ARGS.error
#                 for ko in ARGS.Ko
#                 for m in ARGS.M
#                 for i in range(ARGS.N_exp)
#             ],
#         )
