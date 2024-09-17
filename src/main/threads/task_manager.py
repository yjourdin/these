from collections import defaultdict, deque
from multiprocessing.connection import Connection, wait
from typing import cast

from ...constants import SENTINEL
from ..task import Task


def task_manager(
    succeed: defaultdict[Task, list[Task]],
    precede: defaultdict[Task, list[Task]],
    priority_succeed: defaultdict[Task, set[Task]],
    start: list[Task],
    worker_connections: list[Connection],
    stop_connection: Connection
):
    connections_waiting: list[Connection] = worker_connections.copy()
    connections = worker_connections + [stop_connection]
    to_do_tasks: set[Task] = set(start)
    working_tasks: set[Task] = set()
    done_tasks: set[Task] = set()
    task_deque: deque[Task] = deque(to_do_tasks)

    stop = False
    while (not stop) and connections and (to_do_tasks or working_tasks):

        # Assign a task to a worker
        while task_deque and connections_waiting:
            connection = connections_waiting.pop()
            task = task_deque.pop()
            to_do_tasks.remove(task)
            working_tasks.add(task)
            connection.send(task)

        # Wait for tasks to be done
        for connection in cast(list[Connection], wait(connections)):
            try:
                obj = connection.recv()
            except EOFError:
                connections.remove(connection)
            else:
                if obj == SENTINEL:
                    stop = True
                    break
                else:
                    task = cast(Task, obj)
                    connections_waiting.append(connection)
                    working_tasks.remove(task)
                    done_tasks.add(task)

                    # Add next tasks to task deque
                    for next_task in succeed[task]:
                        if next_task not in to_do_tasks:
                            if all(t in done_tasks for t in precede[next_task]):
                                to_do_tasks.add(next_task)
                                if next_task in priority_succeed[task]:
                                    task_deque.append(next_task)
                                else:
                                    task_deque.appendleft(next_task)

    for connection in connections:
        connection.send(SENTINEL)
