from collections.abc import Mapping
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from queue import LifoQueue
from typing import Any, NamedTuple

from src.constants import SENTINEL_TYPE

from .task import Task, TaskResult

type Args = Mapping[str, Any]


class WorkerArguments(NamedTuple):
    task: Task
    args: Args


class WorkerResult(NamedTuple):
    task: Task
    result: TaskResult


# Stop connections

type StopEndStopConnection = Connection[SENTINEL_TYPE, SENTINEL_TYPE]
type ManagerEndStopConnection = Connection[SENTINEL_TYPE, SENTINEL_TYPE]


class StopConnections(NamedTuple):
    stop_end: StopEndStopConnection
    manager_end: ManagerEndStopConnection


def StopPipe():
    return StopConnections._make(Pipe())


# Task connections

type ThreadEndTaskConnection = Connection[None, TaskResult]
type ManagerEndTaskConnection = Connection[TaskResult | SENTINEL_TYPE, None]


class TaskConnections(NamedTuple):
    thread_end: ThreadEndTaskConnection
    manager_end: ManagerEndTaskConnection


def TaskPipe():
    return TaskConnections._make(Pipe(False))


# Worker connections

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


# Stop event
type StopEvent = Event
