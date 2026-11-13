from collections.abc import Sequence
from concurrent.futures import Future
from multiprocessing.connection import wait
from typing import Any, cast

from src.constants import SENTINEL

from ..connection import TaskPipe, TaskQueueElement, ThreadEndTaskConnection
from ..dir import DIR
from ..task import Task, TaskException, TaskResult, result_list
from .task_manager import TASK_QUEUE


def task_thread(
    task: Task,
    nb_cpus: int = 1,
    precede_futures: list[Future] | None = None,
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
    TASK_QUEUE.put(TaskQueueElement(task, nb_cpus, manager_connection, kwargs))

    result = thread_connection.recv()
    TASK_QUEUE.task_done()

    if result == SENTINEL:
        raise TaskException(str(task))
    return result


def group_task_thread(
    tasks: Sequence[Task],
    nb_cpus: int = 1,
    precede_futures: list[Future] | None = None,
    kwargs: Sequence[dict[str, Any]] | None = None,
):
    kwargs = kwargs or len(tasks) * [{}]
    if all(task.done(DIR, **kw) for task, kw in zip(tasks, kwargs)):
        return TaskResult(None, 0)

    if precede_futures:
        try:
            result_list(precede_futures)
        except Exception:  # noqa: BLE001
            raise TaskException(str(tasks[0]))

    elements: list[TaskQueueElement] = []
    thread_connections: list[ThreadEndTaskConnection] = []

    for task, kw in zip(tasks, kwargs):
        thread_connection, manager_connection = TaskPipe()
        elements.append(TaskQueueElement(task, nb_cpus, manager_connection, kw))
        thread_connections.append(thread_connection)

    TASK_QUEUE.put(elements)

    res = True
    time = 0
    n = 0
    while n < len(tasks):
        for conn in cast(list[ThreadEndTaskConnection], wait(thread_connections)):
            n += 1
            result = conn.recv()
            if result == SENTINEL:
                raise TaskException(str(tasks[thread_connections.index(conn)]))
            res &= result.res
            time = max(time, result.time)

    TASK_QUEUE.task_done()

    return TaskResult(res, time)
