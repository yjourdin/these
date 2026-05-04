import logging
from threading import Thread

from src.constants import SENTINEL

from ..logging import LoggingQueue


class LoggerThread(Thread):
    def __init__(self, queue: LoggingQueue) -> None:
        super().__init__(name="Logging")
        self.queue = queue
        self.start()

    def run(self):
        for record in iter(self.queue.get, SENTINEL):
            logger = logging.getLogger(record.name)
            logger.handle(record)
