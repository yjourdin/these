import logging
import logging.handlers
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import cast

from ..constants import SENTINEL
from .directory import Directory
from .task import Task


def worker(
    dir: Directory,
    connection: Connection,
    logging_queue: Queue,
    stop_error: bool,
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    logger = logging.getLogger("log")

    logger.info("Start")

    for task in iter(connection.recv, SENTINEL):
        try:
            task = cast(Task, task)
            if not task.already_done(dir):
                logger.info("start " + str(task))
                task(dir)
                logger.info("end   " + str(task))
            else:
                logger.info("done  " + str(task))

            connection.send(task)
        except Exception as e:
            logger.error(e, exc_info=True)
            if stop_error:
                connection.send(SENTINEL)

    logger.info("Kill")
