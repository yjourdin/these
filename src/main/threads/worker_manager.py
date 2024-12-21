from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from multiprocessing.connection import Connection, wait
from queue import Empty, LifoQueue
from typing import Any, cast

from ...constants import SENTINEL, SENTINEL_TYPE
from ..task import Task

type TaskQueue = LifoQueue[tuple[Task, Mapping[str, Any], Connection] | SENTINEL_TYPE]


def worker_manager(
    worker_connections: Iterable[Connection],
    task_queue: TaskQueue,
    stop_connection: Connection,
    thread_pool: ThreadPoolExecutor,
    stop_error: bool,
):
    connections = set(worker_connections) | {stop_connection}
    waiting_connections: set[Connection] = set(worker_connections)
    working_connections: dict[Connection, Connection] = {}

    stop = False
    while (not stop) and connections:
        # Get a task from the task queue and assign it to a waiting worker
        empty = False
        while (not stop) and waiting_connections and (not empty):
            if working_connections:
                timeout = 1
            else:
                timeout = None

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
                    connection.send((task, args))

        # Wait a worker to finish a task
        if (not stop) and working_connections:
            if waiting_connections:
                timeout = 1
            else:
                timeout = None

            for connection in cast(list[Connection], wait(connections)):
                try:
                    obj = connection.recv()
                except EOFError:
                    connections.remove(connection)
                else:
                    if obj == SENTINEL:
                        if (connection == stop_connection) or stop_error:
                            stop = True
                            break
                    else:
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
            obj = task_queue.get(timeout=1)
            if obj != SENTINEL:
                task, args, thread_connection = obj
                thread_connection.send(SENTINEL)
    except Empty:
        pass
