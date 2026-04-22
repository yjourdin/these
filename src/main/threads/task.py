from typing import Any

from src.constants import SENTINEL
from src.utils import CustomException

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
        task_queue.put(
            TaskQueueElement(task, args.pop("nb_cpus", 1), args, manager_connection)
        )

        result = thread_connection.recv()

        if result == SENTINEL:
            raise CustomException("Task error")
        return result
    else:
        print(task, args)
