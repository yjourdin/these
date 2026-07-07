from typing import Any

from src.constants import SENTINEL

from ..connection import TaskPipe, TaskQueueElement
from ..dir import DIR
from ..task import FutureTask, Task, TaskException, TaskResult, result_list
from .task_manager import TASK_QUEUE


def task_thread(
    task: Task,
    args: dict[str, Any],
    precede_futures: list[FutureTask],
):
    if task.done(DIR, **args):
        return TaskResult(None, 0)

    try:
        result_list(precede_futures)
    except Exception:  # noqa: BLE001
        raise TaskException(str(task))

    thread_connection, manager_connection = TaskPipe()
    TASK_QUEUE.put(
        TaskQueueElement(task, args.pop("nb_cpus", 1), args, manager_connection)
    )

    result = thread_connection.recv()
    TASK_QUEUE.task_done()

    if result == SENTINEL:
        raise TaskException(str(task))
    return result
