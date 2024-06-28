import csv
import logging
import logging.handlers
from collections import defaultdict
from multiprocessing import JoinableQueue, Queue
from pathlib import Path

from .fieldnames import FIELDNAMES
from .directory import Directory
from .task import Task


def worker(
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
    logging_queue: Queue,
    dir: Directory,
    file_queues: dict[str, Queue],
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)

    logger = logging.getLogger("log")
    for task in iter(task_queue.get, "STOP"):
        try:
            logger.info("start " + str(task))
            task(dir, file_queues)
            done_queue.put(task)
            done_queue.join()
            logger.info("end   " + str(task))
            task_queue.task_done()
        except Exception as e:
            logger.error(e, exc_info=True)
            break

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
    for task in iter(done_queue.get, "STOP"):
        done_set.add(task)
        for next_task in succeed[task]:
            if next_task not in task_set:
                if all(t in done_set for t in precede[next_task]):
                    task_queue.put(next_task)
                    task_set.add(next_task)
        done_queue.task_done()
    done_queue.task_done()


def csv_file_thread(file: Path, q: Queue):
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, FIELDNAMES[file.stem], dialect="unix")
        for result in iter(q.get, "STOP"):
            writer.writerow(result)
            f.flush()
