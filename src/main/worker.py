import csv
import logging
import logging.handlers
from collections import defaultdict
from multiprocessing import JoinableQueue, Queue
from multiprocessing.synchronize import Event
from pathlib import Path

from ..utils import dict_values_to_str
from .directory import Directory
from .fieldnames import FIELDNAMES
from .task import Task

SENTINEL = "STOP"


def worker(
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
    logging_queue: Queue,
    stop_event: Event,
    dir: Directory,
    file_queues: dict[str, Queue],
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    logger = logging.getLogger("log")

    logger.info("Start")
    for task in iter(task_queue.get, SENTINEL):
        try:
            logger.info("start " + str(task))
            task(dir, file_queues)
            logger.info("end   " + str(task))
            done_queue.put(task)
            done_queue.join()
        except Exception as e:
            logger.error(e, exc_info=True)
            if not stop_event.is_set():
                stop_event.set()
        if not stop_event.is_set():
            task_queue.task_done()

    logger.info("Kill")
    task_queue.task_done()


def task_manager(
    succeed: defaultdict[Task, list[Task]],
    precede: defaultdict[Task, list[Task]],
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
):
    task_set = set()
    done_set = set()
    for task in iter(done_queue.get, SENTINEL):
        done_set.add(task)
        for next_task in succeed[task]:
            if next_task not in task_set:
                if all(t in done_set for t in precede[next_task]):
                    task_queue.put(next_task)
                    task_set.add(next_task)
        done_queue.task_done()
    done_queue.task_done()


def logger_thread(q):
    for record in iter(q.get, SENTINEL):
        logger = logging.getLogger(record.name)
        logger.handle(record)


def csv_file_thread(file: Path, q: Queue):
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, FIELDNAMES[file.stem], dialect="unix")
        for result in iter(q.get, SENTINEL):
            writer.writerow(dict_values_to_str(result))
            f.flush()


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
