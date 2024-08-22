from multiprocessing import JoinableQueue
from multiprocessing.synchronize import Event
from pathlib import Path

from ..task import Task


def stopping_thread(event: Event, file: Path, task_queue: "JoinableQueue[Task]"):
    while not event.is_set() and file.exists():
        pass
    if not event.is_set():
        event.set()
    if file.exists():
        file.unlink()
    while not task_queue.empty():
        task_queue.get()
        task_queue.task_done()
    try:
        while True:
            task_queue.task_done()
    except ValueError:
        pass
