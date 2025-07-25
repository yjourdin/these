from typing import Any

from ...constants import SENTINEL
from ..connection import TaskPipe, TaskQueueElement
from ..directory import Directory
from ..task import FutureTaskException, Task, wait_exception_iterable
from .worker_manager import TaskQueue


def task_thread(
    task: Task,
    args: dict[str, Any],
    task_queue: TaskQueue,
    precede_futures: list[FutureTaskException],
    dir: Directory,
):
    if not task.done(dir, **args):
        wait_exception_iterable(precede_futures)

        thread_connection, manager_connection = TaskPipe()
        task_queue.put(TaskQueueElement(task, args, manager_connection))

        result = thread_connection.recv()

        if result == SENTINEL:
            raise Exception("Task error")
        return result
    else:
        print(task, args)
