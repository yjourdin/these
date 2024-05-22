from collections import defaultdict
from multiprocessing import JoinableQueue, Queue
from typing import Any, Literal, cast

from numpy.random import default_rng

from model import ModelType
from run.config import SAConfig

from .arguments import ConfigDict
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


def task_A_train(m: int, n: int, Atr_id: int) -> Task:
    return ("A_train", m, n, Atr_id)


def task_A_test(m: int, n: int, Ate_id: int) -> Task:
    return ("A_test", m, n, Ate_id)


def task_Mo(m: int, model: ModelType, k: int, Mo_id: int) -> Task:
    return ("Mo", m, model, k, Mo_id)


def task_D(
    m: int, n_tr: int, Atr_id: int, Mo: ModelType, ko: int, Mo_id: int, n: int, e: float
) -> Task:
    return ("D", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e)


def task_SA(
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
    config: int,
) -> Task:
    return ("SA", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, Me, ke, config)


def task_MIP(
    m: int,
    n_tr: int,
    Atr_id: int,
    Mo: ModelType,
    ko: int,
    Mo_id: int,
    n: int,
    e: float,
    ke: int,
) -> Task:
    return ("MIP", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, ke)


def task_test(
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
    method: Literal["MIP", "SA"],
    config: int,
    n_te: int,
    Ate_id: int,
) -> Task:
    return (
        "Test",
        m,
        n_tr,
        Atr_id,
        Mo,
        ko,
        Mo_id,
        n,
        e,
        Me,
        ke,
        method,
        config,
        n_te,
        Ate_id,
    )


class TaskExecutor:
    def __init__(
        self,
        args: Arguments,
        dir: Directory,
        seeds: dict[str, list[int]],
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
            case ("A_train", m, n, Atr_id):
                return f"A_train (M: {m:2} N_tr: {n:4} Atr_No: {Atr_id:2})"
            case ("A_test", m, n, Ate_id):
                return f"A_test  (M: {m:2} N_te: {n:4} Ate_No: {Ate_id:2})"
            case ("Mo", m, model, k, Mo_id):
                return (
                    f"Mo      ("
                    f"M: {m:2} "
                    "           "
                    "           "
                    f"Mo: {model:4} "
                    f"Ko: {k:2} "
                    f"Mo_No: {Mo_id:2})"
                )
            case ("D", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e):
                return (
                    f"D       ("
                    f"M: {m:2} "
                    f"N_tr: {n_tr:4} "
                    f"Atr_No: {Atr_id:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"Mo_No: {Mo_id:2} "
                    f"N: {n:4} "
                    f"Error: {e:4})"
                )
            case ("SA", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, Me, ke, config):
                return (
                    f"SA      ("
                    f"M: {m:2} "
                    f"N_tr: {n_tr:4} "
                    f"Atr_No: {Atr_id:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"Mo_No: {Mo_id:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: {Me:4} "
                    f"Ke: {ke:2} "
                    f"Config: {config:4})"
                )
            case ("MIP", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, ke):
                return (
                    f"SA      ("
                    f"M: {m:2} "
                    f"N_tr: {n_tr:4} "
                    f"Atr_No: {Atr_id:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"Mo_No: {Mo_id:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: SRMP "
                    f"Ke: {ke:2})"
                )
            case (
                "Test",
                m,
                n_tr,
                Atr_id,
                Mo,
                ko,
                Mo_id,
                n,
                e,
                Me,
                ke,
                method,
                config,
                n_te,
                Ate_id,
            ):
                config = config or 0
                return (
                    f"Test    ("
                    f"M: {m:2} "
                    f"N_tr: {n_tr:4} "
                    f"Atr_No: {Atr_id:2} "
                    f"Mo: {Mo:4} "
                    f"Ko: {ko:2} "
                    f"Mo_No: {Mo_id:2} "
                    f"N: {n:4} "
                    f"Error: {e:4} "
                    f"Me: SRMP "
                    f"Ke: {ke:2})"
                    f"Config: {config:4} "
                    f"Method: {method:3} "
                    f"N_te: {n_te:4} "
                    f"Ate_No: {Ate_id:2})"
                )
            case _:
                raise ValueError("Unknown task")

    def execute(self, task: Task):
        match task:
            case ("A_train", m, n, Atr_id):
                create_A_train(
                    m,
                    n,
                    Atr_id,
                    self.dir,
                    default_rng([n, m, self.seeds["A_train"][Atr_id]]),
                )
            case ("A_test", m, n, Ate_id):
                create_A_test(
                    m,
                    n,
                    Ate_id,
                    self.dir,
                    default_rng([n, m, self.seeds["A_test"][Ate_id]]),
                )
            case ("Mo", m, model, k, Mo_id):
                create_Mo(
                    m,
                    model,
                    k,
                    Mo_id,
                    self.dir,
                    default_rng([k, m, self.seeds["Mo"][Mo_id]]),
                )
            case ("D", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e):
                create_D(
                    m,
                    n_tr,
                    Atr_id,
                    Mo,
                    ko,
                    Mo_id,
                    n,
                    e,
                    self.dir,
                    default_rng(
                        [
                            n,
                            self.seeds["Mo"][Mo_id],
                            ko,
                            self.seeds["A_train"][Atr_id],
                            n_tr,
                            m,
                        ]
                    ),
                )
            case ("SA", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, Me, ke, config):
                time, it, best_fitness = run_SA(
                    m,
                    n_tr,
                    Atr_id,
                    Mo,
                    ko,
                    Mo_id,
                    n,
                    e,
                    Me,
                    ke,
                    config,
                    cast(SAConfig, self.args.config["SA"][config]),
                    self.dir,
                    default_rng(
                        [
                            ke,
                            n,
                            self.seeds["Mo"][Mo_id],
                            ko,
                            self.seeds["A_train"][Atr_id],
                            n_tr,
                            m,
                        ]
                    ),
                )
                self.train_results_queue.put(
                    [
                        m,
                        n_tr,
                        Atr_id,
                        Mo,
                        ko,
                        Mo_id,
                        n,
                        e,
                        Me,
                        ke,
                        "SA",
                        config,
                        time,
                        best_fitness,
                        it,
                    ]
                )
            case ("MIP", m, n_tr, Atr_id, Mo, ko, Mo_id, n, e, ke):
                time, best_fitness = run_MIP(
                    m,
                    n_tr,
                    Atr_id,
                    Mo,
                    ko,
                    Mo_id,
                    n,
                    e,
                    ke,
                    self.dir,
                    default_rng(
                        [
                            ke,
                            n,
                            self.seeds["Mo"][Mo_id],
                            ko,
                            self.seeds["A_train"][Atr_id],
                            n_tr,
                            m,
                        ]
                    ).integers(2_000_000_000),
                )
                self.train_results_queue.put(
                    [
                        m,
                        n_tr,
                        Atr_id,
                        Mo,
                        ko,
                        Mo_id,
                        n,
                        e,
                        "SRMP",
                        ke,
                        "MIP",
                        "",
                        time,
                        best_fitness,
                        "",
                    ]
                )
            case (
                "Test",
                m,
                n_tr,
                Atr_id,
                Mo,
                ko,
                Mo_id,
                n,
                e,
                Me,
                ke,
                method,
                config,
                n_te,
                Ate_id,
            ):
                test_fitness, kendall_tau = run_test(
                    m,
                    n_tr,
                    Atr_id,
                    Mo,
                    ko,
                    Mo_id,
                    n,
                    e,
                    Me,
                    ke,
                    method,
                    config,
                    n_te,
                    Ate_id,
                    self.dir,
                )
                self.test_results_queue.put(
                    [
                        m,
                        n_tr,
                        Atr_id,
                        Mo,
                        ko,
                        Mo_id,
                        n,
                        e,
                        Me,
                        ke,
                        method,
                        config,
                        n_te,
                        Ate_id,
                        test_fitness,
                        kendall_tau,
                    ]
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
