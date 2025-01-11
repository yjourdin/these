from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from multiprocessing.connection import wait
from queue import Empty
from typing import cast

from ...constants import SENTINEL
from ..connection import (
    ManagerStopConnection,
    ManagerTaskConnection,
    ManagerWorkerConnection,
    TaskQueue,
    WorkerArguments,
)


def worker_manager(
    worker_connections: Iterable[ManagerWorkerConnection],
    task_queue: TaskQueue,
    stop_connection: ManagerStopConnection,
    thread_pool: ThreadPoolExecutor,
    stop_error: bool,
):
    connections = set(worker_connections) | {stop_connection}
    waiting_connections: set[ManagerWorkerConnection] = set(worker_connections)
    working_connections: dict[ManagerWorkerConnection, ManagerTaskConnection] = {}

    stop = False
    while (not stop) and connections:
        # Get a task from the task queue and assign it to a waiting worker
        empty = False
        while (not stop) and waiting_connections and (not empty):
            timeout = 1 if working_connections else None

            try:
                obj = task_queue.get(timeout=timeout)
            except Empty:
                empty = True
            else:
                if obj == SENTINEL:
                    stop = True
                else:
                    task, args, thread_connection = obj
                    connection = waiting_connections.pop()
                    working_connections[connection] = thread_connection
                    connection.send(WorkerArguments(task, args))

        # Wait a worker to finish a task
        if (not stop) and working_connections:
            timeout = 1 if waiting_connections else None

            for connection in wait(connections, timeout=timeout):  # type: ignore
                connection = cast(
                    ManagerWorkerConnection | ManagerStopConnection, connection
                )
                try:
                    obj = connection.recv()
                except EOFError:
                    connections.remove(connection)
                else:
                    if obj == SENTINEL and (
                        (connection == stop_connection) or stop_error
                    ):
                        stop = True
                        break
                    else:
                        connection = cast(ManagerWorkerConnection, connection)
                        working_connections[connection].send(obj)
                        del working_connections[connection]
                        waiting_connections.add(connection)

    # Shutdown threadpool
    thread_pool.shutdown(False, cancel_futures=True)

    # Stop all workers connections and working threads
    for connection in chain(connections, working_connections.values()):
        connection.send(SENTINEL)

    # Stop waiting threads
    try:
        while True:
            if (obj := task_queue.get(timeout=1)) != SENTINEL:
                task, args, thread_connection = obj
                thread_connection.send(SENTINEL)
    except Empty:
        pass
