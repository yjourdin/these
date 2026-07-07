from contextlib import suppress
from itertools import chain
from multiprocessing.connection import wait
from queue import Empty, LifoQueue, ShutDown
from threading import Thread
from typing import cast

from src.constants import SENTINEL, SENTINEL_TYPE

from ..connection import (
    ManagerEndTaskConnection,
    ManagerEndWorkerConnection,
    TaskQueue,
    TaskQueueElement,
    WorkerArguments,
    WorkerResult,
)
from ..task import Task, TaskResult
from .stop import STOP

TASK_QUEUE: TaskQueue = LifoQueue()


class TaskManager(Thread):
    def __init__(
        self, connections: list[ManagerEndWorkerConnection], stop_on_error: bool
    ):
        super().__init__(name="Task manager")
        self.worker_connections = {
            worker: connection for worker, connection in enumerate(connections)
        }
        self.waiting = set(self.worker_connections.keys())
        self.task_connections: dict[Task, ManagerEndTaskConnection] = {}
        self.working: dict[Task, list[int]] = {}
        self.stop_on_error = stop_on_error
        self.start()

    def send_task(self, element: TaskQueueElement):
        task, nb_cpus, args, connection = element
        self.task_connections[task] = connection

        if len(self.worker_connections) < nb_cpus:
            self.send_result(task)
        elif len(self.waiting) < nb_cpus:
            return False
        else:
            workers = [self.waiting.pop() for _ in range(nb_cpus)]
            self.working[task] = workers
            self.worker_connections[workers[0]].send(WorkerArguments(task, args))

        return True

    def receive_result(self, connection: ManagerEndWorkerConnection):
        try:
            return connection.recv()
        except EOFError:
            worker = -1
            for worker, c in self.worker_connections.items():
                if c == connection:
                    del self.worker_connections[worker]
                    break
            if worker != -1:
                for task, workers in self.working.items():
                    if worker in workers:
                        self.working[task].remove(worker)
                        return WorkerResult(task, SENTINEL)

    def send_result(self, task: Task, result: TaskResult | SENTINEL_TYPE = SENTINEL):
        self.task_connections.pop(task).send(result)

    def stop(self):
        # Shutdown task queue
        TASK_QUEUE.shutdown()

        # Get remaining tasks
        with suppress(ShutDown):
            while True:
                task, _, _, connection = TASK_QUEUE.get()
                self.task_connections[task] = connection

        # Send sentinel signal
        for connection in chain(
            self.task_connections.values(), self.worker_connections.values()
        ):
            connection.send(SENTINEL)

    def run(self):
        to_do = None

        while (self.waiting or self.working) and not STOP.is_set():
            while self.waiting:
                try:
                    element = to_do or TASK_QUEUE.get(timeout=1)
                except Empty:
                    break
                except ShutDown:
                    STOP.set()
                else:
                    # Send task
                    if to_do := None if self.send_task(element) else element:
                        break

            if self.working:
                for connection in cast(
                    list[ManagerEndWorkerConnection],
                    wait(self.worker_connections.values(), timeout=1),
                ):
                    if obj := self.receive_result(connection):
                        task, result = obj
                        if self.stop_on_error and (result == SENTINEL):
                            STOP.set()
                        self.send_result(task, result)
                        self.waiting |= set(self.working.pop(task))

        STOP.set()
        self.stop()
