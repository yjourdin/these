from pathlib import Path
from time import sleep

from ...constants import SENTINEL
from ..connection import StopStopConnection
from .worker_manager import TaskQueue


def stopping_thread(file: Path, connection: StopStopConnection, task_queue: TaskQueue):
    while file.exists() and not connection.poll():
        sleep(1)
    if file.exists():
        file.unlink()
    if connection.poll():
        connection.recv()
    else:
        connection.send(SENTINEL)
        task_queue.put(SENTINEL)
