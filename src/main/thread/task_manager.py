from collections import defaultdict
from multiprocessing import JoinableQueue
from multiprocessing.connection import Connection, wait
from multiprocessing.synchronize import Event
from typing import cast

from ...constants import SENTINEL
from ..task import Task


def task_manager(
    succeed: defaultdict[Task, list[Task]],
    precede: defaultdict[Task, list[Task]],
    follow_up: dict[Task, Task],
    task_queue: "JoinableQueue[Task]",
    connections: list[Connection],
    stop_event: Event,
):
    task_set = set()
    done_set = set()
    while not stop_event.is_set():
        for connection in wait(connections):
            connection = cast(Connection, connection)
            try:
                task = connection.recv()
            except EOFError:
                pass
            else:
                done_set.add(task)
                sent = False
                for next_task in succeed[task]:
                    if next_task not in task_set:
                        if all(t in done_set for t in precede[next_task]):
                            task_set.add(next_task)
                            if follow_up.get(task) == next_task:
                                connection.send(next_task)
                                sent = True
                            else:
                                task_queue.put(next_task)
                if not sent:
                    connection.send(SENTINEL)
