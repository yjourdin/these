import logging
import logging.handlers

from src.constants import SENTINEL

from .connection import ProcessEndWorkerConnection, WorkerResult
from .directory import Directory
from .logging import LoggingQueue


def worker(
    connection: ProcessEndWorkerConnection,
    logging_queue: LoggingQueue,
    dir: Directory,
):
    # Logging setup
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    logger = logging.getLogger("log")

    logger.info("Start")

    for task, args in iter(connection.recv, SENTINEL):
        try:
            logger.info(f"{'start':5} {task!s}")
            connection.send(WorkerResult(task, task(dir, **args)))
            logger.info(f"{'end':5} {task!s}")
        except Exception:
            logger.exception("Task error")
            connection.send(SENTINEL)

    logger.info("Kill")
