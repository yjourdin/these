from collections import defaultdict
from multiprocessing import JoinableQueue, Queue
from typing import Any, Literal

from numpy.random import default_rng

from model import ModelType

from .arguments import Arguments
from .job import (
    create_A_test,
    create_A_train,
    create_D,
    create_Mo,
    run_MIP,
    run_SA,
    run_test,
)
from .path import Directory

Task = tuple[Any, ...]


def task_A_train(i: int, n: int, m: int) -> Task:
    return ("A_train", i, n, m)


def task_A_test(i: int, n: int, m: int) -> Task:
    return ("A_test", i, n, m)


def task_Mo(i: int, m: int, model: ModelType, k: int) -> Task:
    return ("Mo", i, m, model, k)


def task_D(i: int, n_tr: int, m: int, Mo: ModelType, ko: int, n: int, e: float) -> Task:
    return ("D", i, n_tr, m, Mo, ko, n, e)


def task_SA(
    i: int,
    n_tr: int,
    m: int,
    Mo: ModelType,
    ko: int,
    n: int,
    e: float,
    Me: ModelType,
    ke: int,
    config: int,
) -> Task:
    return ("SA", i, n_tr, m, Mo, ko, n, e, Me, ke, config)


def task_MIP(
    i: int,
    n_tr: int,
    m: int,
    Mo: ModelType,
    ko: int,
    n: int,
    e: float,
    ke: int,
) -> Task:
    return ("MIP", i, n_tr, m, Mo, ko, n, e, ke)


def task_test(
    i: int,
    n_tr: int,
    n_te: int,
    m: int,
    Mo: ModelType,
    ko: int,
    n: int,
    e: float,
    Me: ModelType,
    ke: int,
    method: Literal["MIP", "SA"],
    config: int | None = None,
) -> Task:
    return ("Test", i, n_tr, n_te, m, Mo, ko, n, e, Me, ke, method, config)


class TaskExecutor:
    def __init__(
        self,
        args: Arguments,
        dir: Directory,
        seeds: list[int],
        train_results_queue: Queue,
        test_results_queue: Queue,
    ) -> None:
        self.args = args
        self.dir = dir
        self.seeds = seeds
        self.train_results_queue = train_results_queue
        self.test_results_queue = test_results_queue

    def name(self, task: Task):
        match task:
            case ("A_train", i, n, m):
                return f"A_train (DM: {i:2} N_tr: {n:4} M: {m:2})"
            case ("A_test", i, n, m):
                return f"A_test  (DM: {i:2} N_te: {n:4} M: {m:2})"
            case ("Mo", i, m, model, k):
                return (
                    f"Mo      ("
                    f"DM: {i:2} "
                    "           "
                    f"M: {m:2} "
                    f"Mo: {model:4} "
                    f"Ko: {k:2})"
                )
            case ("D", i, n_tr, m, Mo, ko, n, e):
                return (
                    f"D       ("
                    f"DM: {i:2} "
                    f"N_tr: {n_tr:4} "
                    f"M: {m:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"N: {n:4} "
                    f"Error: {e:4})"
                )
            case ("SA", i, n_tr, m, Mo, ko, n, e, Me, ke, config):
                return (
                    f"SA      ("
                    f"DM: {i:2} "
                    f"N_tr: {n_tr:4} "
                    f"M: {m:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: {Me:4} "
                    f"Ke: {ke:2} "
                    f"Config: {config:4})"
                )
            case ("MIP", i, n_tr, m, Mo, ko, n, e, ke):
                return (
                    f"MIP     ("
                    f"DM: {i:2} "
                    f"N_tr: {n_tr:4} "
                    f"M: {m:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: SRMP "
                    f"Ke: {ke:2})"
                )
            case ("Test", i, n_tr, n_te, m, Mo, ko, n, e, Me, ke, method, config):
                config = config or 0
                return (
                    f"Test    ("
                    f"DM: {i:2} "
                    f"N_tr: {n_tr:4} "
                    f"M: {m:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: {Me:4} "
                    f"Ke: {ke:2} "
                    f"Config: {config:4} "
                    f"Method: {method:3} "
                    f"N_te: {n_te:4})"
                )
            case _:
                raise ValueError("Unknown task")

    def execute(self, task: Task):
        match task:
            case ("A_train", i, n, m):
                create_A_train(n, m, i, self.dir, default_rng([n, m, self.seeds[i]]))
            case ("A_test", i, n, m):
                create_A_test(n, m, i, self.dir, default_rng([n, m, self.seeds[i]]))
            case ("Mo", i, m, model, k):
                create_Mo(model, k, m, i, self.dir, default_rng([k, m, self.seeds[i]]))
            case ("D", i, n_tr, m, Mo, ko, n, e):
                create_D(
                    n,
                    e,
                    Mo,
                    ko,
                    m,
                    n_tr,
                    i,
                    self.dir,
                    default_rng([n, ko, m, self.seeds[i]]),
                )
            case ("SA", i, n_tr, m, Mo, ko, n, e, Me, ke, config):
                time, it, best_fitness = run_SA(
                    config,
                    Me,
                    ke,
                    n,
                    e,
                    Mo,
                    ko,
                    m,
                    n_tr,
                    i,
                    self.args.config[config].T0[n],
                    self.args.config[config].alpha,
                    self.args.config[config].amp,
                    self.args.config[config].max_iter,
                    self.dir,
                    default_rng([ke, n, ko, m, self.seeds[i]]),
                )
                self.train_results_queue.put(
                    f"{i},"
                    f"{n_tr},"
                    f"{m},"
                    f"{Mo},"
                    f"{ko},"
                    f"{n},"
                    f"{e},"
                    f"{Me},"
                    f"{ke},"
                    f"SA,"
                    f"{config},"
                    f"{time},"
                    f"{best_fitness},"
                    f"{it}\n"
                )
            case ("MIP", i, n_tr, m, Mo, ko, n, e, ke):
                time, best_fitness = run_MIP(
                    ke,
                    n,
                    e,
                    Mo,
                    ko,
                    m,
                    n_tr,
                    i,
                    self.dir,
                    default_rng([ke, n, ko, m, self.seeds[i]]).integers(2_000_000_000),
                )
                self.train_results_queue.put(
                    f"{i},"
                    f"{n_tr},"
                    f"{m},"
                    f"{Mo},"
                    f"{ko},"
                    f"{n},"
                    f"{e},"
                    f"SRMP,"
                    f"{ke},"
                    f"MIP,,"
                    f"{time},"
                    f"{best_fitness},,\n"
                )
            case ("Test", i, n_tr, n_te, m, Mo, ko, n, e, Me, ke, method, config):
                test_fitness, kendall_tau = run_test(
                    config, method, Me, ke, n, e, Mo, ko, m, n_te, n_tr, i, self.dir
                )
                self.test_results_queue.put(
                    f"{i},"
                    f"{n_tr},"
                    f"{n_te},"
                    f"{m},"
                    f"{Mo},"
                    f"{ko},"
                    f"{n},"
                    f"{e},"
                    f"{Me},"
                    f"{ke},"
                    f"{method},"
                    f"{config},"
                    f"{test_fitness},"
                    f"{kendall_tau}\n"
                )
            case _:
                raise ValueError("Unknown task")


def task_manager(
    succeed: defaultdict[Task, list[Task]],
    precede: defaultdict[Task, list[Task]],
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
):
    task_set = set()
    done_set = set()
    for task in iter(done_queue.get, "STOP"):
        done_set.add(task)
        for next_task in succeed[task]:
            if next_task not in task_set:
                if all(t in done_set for t in precede[next_task]):
                    task_queue.put(next_task)
                    task_set.add(next_task)
        done_queue.task_done()
    done_queue.task_done()
