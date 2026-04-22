from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import wait
from queue import Empty
from typing import NamedTuple, cast

from src.constants import SENTINEL

from ..connection import (
    ManagerEndTaskConnection,
    ManagerEndWorkerConnection,
    StopEvent,
    TaskQueue,
    TaskQueueElement,
    WorkerArguments,
)
from ..task import Task


class WorkingTask(NamedTuple):
    connection: ManagerEndTaskConnection
    workers: set[int]


def worker_manager(
    worker_connections: list[ManagerEndWorkerConnection],
    task_queue: TaskQueue,
    stop: StopEvent,
    thread_pool: ThreadPoolExecutor,
    stop_on_error: bool,
):
    waiting: set[int] = set(range(len(worker_connections)))
    working_tasks: dict[Task, WorkingTask] = {}

    to_do: TaskQueueElement | None = None

    while (not stop.is_set()) and worker_connections:
        # Get a task from the task queue and assign it to a waiting worker
        while (
            (not stop.is_set())
            and waiting
        ):
            if (not to_do) or (len(worker_connections) < to_do.nb_cpus):
                try:
                    to_do = task_queue.get(timeout=1)
                except Empty:
                    to_do = None
                    break

            if to_do and (to_do.nb_cpus <= len(waiting)):
                task, nb_cpus, args, connection = to_do

                workers = {waiting.pop() for _ in range(nb_cpus)}
                working_tasks[task] = WorkingTask(connection, workers)
                worker_connections[next(iter(workers))].send(WorkerArguments(task, args))

                to_do = None

        # Wait a worker to finish a task
        if (not stop.is_set()) and working_tasks:
            for connection in cast(
                list[ManagerEndWorkerConnection], wait(worker_connections, timeout=1)
            ):
                try:
                    obj = connection.recv()
                except EOFError:
                    worker_connections.remove(connection)
                else:
                    if obj == SENTINEL and stop_on_error:
                        stop.set()
                        break
                    else:
                        task, result = obj
                        connection, workers = working_tasks.pop(task)

                        connection.send(result)
                        waiting |= workers

    # Shutdown threadpool
    thread_pool.shutdown(False, cancel_futures=True)

    # Stop all workers
    for connection in worker_connections:
        connection.send(SENTINEL)

    # Stop working tasks threads
    for connection, _ in working_tasks.values():
        connection.send(SENTINEL)

    # Stop waiting tasks threads
    while not task_queue.empty():
        try:
            if (obj := task_queue.get(timeout=1)) != SENTINEL:
                obj.connection.send(SENTINEL)
        except Empty:
            continue
