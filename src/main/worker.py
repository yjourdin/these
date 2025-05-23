import logging
import logging.handlers

from ..constants import SENTINEL
from .connection import ProcessWorkerConnection
from .directory import Directory
from .logging import LoggingQueue


def worker(
    connection: ProcessWorkerConnection, logging_queue: LoggingQueue, dir: Directory
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    logger = logging.getLogger("log")

    logger.info("Start")

    for task, args in iter(connection.recv, SENTINEL):
        try:
            logger.info("start " + str(task))
            connection.send(task(dir, **args))
            logger.info("end   " + str(task))
        except Exception as e:
            logger.error(e, exc_info=True)
            connection.send(SENTINEL)

    logger.info("Kill")
