from argparse import Namespace
from collections import defaultdict
from multiprocessing import Queue
from multiprocessing.managers import DictProxy
from typing import Any, Literal

from numpy.random import default_rng

from model import ModelType

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
    config: int,
) -> Task:
    return ("Test", i, n_tr, n_te, m, Mo, ko, n, e, Me, ke, method, config)


class TaskManager:
    def __init__(
        self,
        args: Namespace,
        succeed: defaultdict[Task, list[Task]],
        precede: defaultdict[Task, list[Task]],
        dir: Directory,
        seeds: list[int],
        configs: dict,
        train_results_queue: Queue,
        test_results_queue: Queue,
    ) -> None:
        self.args = args
        self.succeed = succeed
        self.precede = precede
        self.dir = dir
        self.seeds = seeds
        self.configs = configs
        self.train_results_queue = train_results_queue
        self.test_results_queue = test_results_queue

    def name(self, task: Task):
        match task:
            case ("A_train", i, n, m):
                return f"A_train (DM: {i:2} N: {n:4} M: {m:2})"
            case ("A_test", i, n, m):
                return f"A_test  (DM: {i:2} N: {n:4} M: {m:2})"
            case ("Mo", i, m, model, k):
                return f"Mo      (DM: {i:2} M: {m:2} Mo: {model:4} Ko: {k:2})"
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
                return (
                    f"Test    ("
                    f"DM: {i:2} "
                    f"N_tr: {n_tr:4} "
                    f"N_te: {n_te:4} "
                    f"M: {m:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: {Me:4} "
                    f"Ke: {ke:2} "
                    f"Method: {method:3} "
                    f"Config: {config:4})"
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
                    Me,
                    ke,
                    n,
                    e,
                    Mo,
                    ko,
                    m,
                    n_tr,
                    i,
                    self.configs[config]["T0"][n],
                    self.configs[config]["Tf"][n],
                    self.configs[config]["alpha"],
                    self.configs[config]["amp"],
                    self.dir,
                    default_rng([ke, n, ko, m, self.seeds[i]]),
                )
                self.train_results_queue.put(
                    f"{i},"
                    f"{n_tr}"
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
                )
                self.train_results_queue.put(
                    f"{i},"
                    f"{n_tr}"
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
                    Me, ke, n, e, Mo, ko, m, n_te, n_tr, i, self.dir
                )
                self.test_results_queue.put(
                    f"{i},"
                    f"{n_tr}"
                    f"{n_te}"
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

    def next_tasks(self, task: Task, done_dict: "DictProxy[Task, bool]"):
        tasks = []
        for next_task in self.succeed[task]:
            if all(done_dict.get(t, False) for t in self.precede[next_task]):
                tasks.append(next_task)
        return tasks
