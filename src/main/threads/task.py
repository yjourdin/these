from concurrent.futures import FIRST_EXCEPTION, Future, wait
from multiprocessing import Pipe
from typing import Any

from ...constants import SENTINEL
from ..directory import Directory
from ..task import Task
from .worker_manager import TaskQueue


def task_thread(
    task: Task,
    args: dict[str, Any],
    task_queue: TaskQueue,
    precede_futures: list[Future],
    dir: Directory,
):
    if not task.done(dir, **args):
        done, not_done = wait(precede_futures, return_when=FIRST_EXCEPTION)
        for future in done:
            err = future.exception()
            if err is not None:
                raise err

        thread_connection, worker_connection = Pipe()
        task_queue.put((task, args, worker_connection))

        result = thread_connection.recv()
        if result == SENTINEL:
            raise Exception()
        return result
