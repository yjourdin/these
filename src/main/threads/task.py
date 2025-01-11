from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import Any, cast

from ...constants import SENTINEL
from ..connection import TaskPipe, TaskQueueElement
from ..directory import Directory
from ..task import Task, raise_exceptions
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

        thread_connection, manager_connection = TaskPipe()
        task_queue.put(TaskQueueElement(task, args, manager_connection))
        
        result = thread_connection.recv()
        if result == SENTINEL:
            raise Exception()
        return result
