import logging
import logging.handlers

from ...constants import SENTINEL


def logger_thread(q):
    for record in iter(q.get, SENTINEL):
        logger = logging.getLogger(record.name)
        logger.handle(record)
