from collections.abc import Mapping
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from queue import LifoQueue
from typing import Any, NamedTuple

from src.constants import SENTINEL_TYPE

from .task import Task, TaskResult

# Task connections

type ThreadEndTaskConnection = Connection[None, TaskResult | SENTINEL_TYPE]
type ManagerEndTaskConnection = Connection[TaskResult | SENTINEL_TYPE, None]


class TaskConnections(NamedTuple):
    thread_end: ThreadEndTaskConnection
    manager_end: ManagerEndTaskConnection


def TaskPipe():
    return TaskConnections._make(Pipe(False))


# Worker connections

type Args = Mapping[str, Any]


class WorkerArguments(NamedTuple):
    task: Task
    args: Args


class WorkerResult(NamedTuple):
    task: Task
    result: TaskResult | SENTINEL_TYPE


type ProcessEndWorkerConnection = Connection[
    WorkerResult | SENTINEL_TYPE, WorkerArguments
]
type ManagerEndWorkerConnection = Connection[
    WorkerArguments | SENTINEL_TYPE, WorkerResult
]


class WorkerConnections(NamedTuple):
    thread_end: ProcessEndWorkerConnection
    manager_end: ManagerEndWorkerConnection


def WorkerPipe():
    return WorkerConnections._make(Pipe())


# Task queue


class TaskQueueElement(NamedTuple):
    task: Task
    nb_cpus: int
    args: Args
    connection: ManagerEndTaskConnection


type TaskQueue = LifoQueue[TaskQueueElement]
