import logging
import logging.handlers
from multiprocessing import Queue
from typing import Any

from ...constants import SENTINEL


def logger_thread(q: "Queue[Any]"):
    for record in iter(q.get, SENTINEL):
        logger = logging.getLogger(record.name)
        logger.handle(record)
