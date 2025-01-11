from collections.abc import Mapping
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from queue import LifoQueue
from typing import Any, NamedTuple

from ..constants import SENTINEL_TYPE
from .task import Task, TaskResult

type Args = Mapping[str, Any]


class WorkerArguments(NamedTuple):
    task: Task
    args: Args


# Stop connections

type StopStopConnection = Connection[SENTINEL_TYPE, SENTINEL_TYPE]
type ManagerStopConnection = Connection[SENTINEL_TYPE, SENTINEL_TYPE]


class StopConnections(NamedTuple):
    stop: StopStopConnection
    manager: ManagerStopConnection


def StopPipe():
    return StopConnections._make(Pipe())


# Task connections

type ThreadTaskConnection = Connection[None, TaskResult]
type ManagerTaskConnection = Connection[TaskResult | SENTINEL_TYPE, None]


class TaskConnections(NamedTuple):
    thread: ThreadTaskConnection
    manager: ManagerTaskConnection


def TaskPipe():
    return TaskConnections._make(Pipe(False))


# Worker connections

type ProcessWorkerConnection = Connection[TaskResult, WorkerArguments]
type ManagerWorkerConnection = Connection[WorkerArguments | SENTINEL_TYPE, TaskResult]


class WorkerConnections(NamedTuple):
    thread: ThreadTaskConnection
    manager: ManagerTaskConnection


def WorkerPipe():
    return WorkerConnections._make(Pipe())


# Task queue


class TaskQueueElement(NamedTuple):
    task: Task
    args: Args
    connection: ManagerTaskConnection


type TaskQueue = LifoQueue[TaskQueueElement | SENTINEL_TYPE]
