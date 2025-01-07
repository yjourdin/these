from concurrent.futures import Future
from multiprocessing import Pipe
from typing import Any

from ...constants import SENTINEL
from ...utils import raise_exceptions
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
        raise_exceptions(precede_futures)

        thread_connection, worker_connection = Pipe()
        task_queue.put((task, args, worker_connection))

        result = thread_connection.recv()
        if result == SENTINEL:
            raise Exception()
        return result
