import logging
import logging.handlers
from multiprocessing import JoinableQueue, Queue
from multiprocessing.synchronize import Event

from .directory import Directory
from .task import Task
from ..constants import SENTINEL


def worker(
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
    logging_queue: Queue,
    stop_event: Event,
    dir: Directory,
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
            task(dir)
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
