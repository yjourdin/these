import logging
import logging.handlers
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, cast

from ..constants import SENTINEL
from .directory import Directory
from .task import Task


def worker(connection: Connection, logging_queue: Queue, dir: Directory):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)
    logger = logging.getLogger("log")

    logger.info("Start")

    for task, args in iter(connection.recv, SENTINEL):
        try:
            task = cast(Task, task)
            args = cast(dict[str, Any], args)
            logger.info("start " + str(task))
            connection.send(task(dir, **args))
            logger.info("end   " + str(task))
        except Exception as e:
            logger.error(e, exc_info=True)
            connection.send(SENTINEL)

    logger.info("Kill")
