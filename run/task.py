from argparse import Namespace
from collections import defaultdict
from multiprocessing import Queue
from multiprocessing.managers import DictProxy
from typing import Any

from numpy.random import Generator

from model import ModelType

from .job import (
    compute_test,
    create_A_test,
    create_A_train,
    create_D,
    create_Me,
    create_Mo,
)
from .path import Directory

Task = tuple[Any, ...]


def task_A_train(i: int, m: int) -> Task:
    return ("A_train", i, m)


def task_A_test(i: int, m: int) -> Task:
    return ("A_test", i, m)


def task_Mo(i: int, m: int, model: ModelType, k: int) -> Task:
    return ("Mo", i, m, model, k)


def task_D(i: int, m: int, Mo: ModelType, ko: int, n: int, e: float) -> Task:
    return ("D", i, m, Mo, ko, n, e)


def task_Me(
    i: int, m: int, Mo: ModelType, ko: int, n: int, e: float, Me: ModelType, ke: int
) -> Task:
    return ("Me", i, m, Mo, ko, n, e, Me, ke)


def task_test(
    i: int, m: int, Mo: ModelType, ko: int, n: int, e: float, Me: ModelType, ke: int
) -> Task:
    return ("Test", i, m, Mo, ko, n, e, Me, ke)


class TaskManager:
    def __init__(
        self,
        args: Namespace,
        succeed: defaultdict[Task, list[Task]],
        precede: defaultdict[Task, list[Task]],
        dir: Directory,
        rngs: list[Generator],
        train_results_queue: Queue,
        test_results_queue: Queue,
    ) -> None:
        self.args = args
        self.succeed = succeed
        self.precede = precede
        self.dir = dir
        self.rngs = rngs
        self.train_results_queue = train_results_queue
        self.test_results_queue = test_results_queue

    def execute(self, task: Task):
        match task:
            case ("A_train", i, m):
                create_A_train(self.args.N_tr, m, i, self.dir, self.rngs[i])
            case ("A_test", i, m):
                create_A_test(self.args.N_te, m, i, self.dir, self.rngs[i])
            case ("Mo", i, m, model, k):
                create_Mo(model, k, m, i, self.dir, self.rngs[i])
            case ("D", i, m, Mo, ko, n, e):
                create_D(n, e, Mo, ko, m, i, self.dir, self.rngs[i])
            case ("Me", i, m, Mo, ko, n, e, Me, ke):
                create_Me(
                    Me,
                    ke,
                    n,
                    e,
                    Mo,
                    ko,
                    m,
                    i,
                    self.args.T0[n],
                    self.args.Tf[n],
                    self.args.alpha,
                    self.dir,
                    self.rngs[i],
                    self.train_results_queue,
                )
            case ("Test", i, m, Mo, ko, n, e, Me, ke):
                compute_test(
                    Me, ke, n, e, Mo, ko, m, i, self.dir, self.test_results_queue
                )
            case _:
                raise ValueError("Unknown task")

    def next_tasks(self, task: Task, done_dict: "DictProxy[Task, bool]"):
        tasks = []
        for next_task in self.succeed[task]:
            if all([done_dict.get(t, False) for t in self.precede[next_task]]):
                tasks.append(next_task)
        return tasks
