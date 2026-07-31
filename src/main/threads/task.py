from typing import Any

from src.constants import SENTINEL

from ..connection import TaskPipe, TaskQueueElement
from ..dir import DIR
from ..task import FutureTask, Task, TaskException, TaskResult, result_list
from .task_manager import TASK_QUEUE


def task_thread(
    task: Task,
    nb_cpus: int = 1,
    precede_futures: list[FutureTask] | None = None,
    **kwargs: Any,
):
    if task.done(DIR, **kwargs):
        return TaskResult(None, 0)

    if precede_futures:
        try:
            result_list(precede_futures)
        except Exception:  # noqa: BLE001
            raise TaskException(str(task))

    thread_connection, manager_connection = TaskPipe()
    TASK_QUEUE.put(
        TaskQueueElement(task, nb_cpus, manager_connection, kwargs)
    )

    result = thread_connection.recv()
    TASK_QUEUE.task_done()

    if result == SENTINEL:
        raise TaskException(str(task))
    return result
