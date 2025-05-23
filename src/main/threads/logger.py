import logging

from ...constants import SENTINEL
from ..logging import LoggingQueue


def logger_thread(q: LoggingQueue):
    for record in iter(q.get, SENTINEL):
        logger = logging.getLogger(record.name)
        logger.handle(record)
