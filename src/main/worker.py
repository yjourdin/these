import logging
import logging.handlers
from multiprocessing import Process

from src.constants import SENTINEL

from .connection import ProcessEndWorkerConnection, WorkerResult
from .directory import Directory
from .logging import LoggingQueue


class WorkerProcess(Process):
    def __init__(
        self,
        connection: ProcessEndWorkerConnection,
        logging_queue: LoggingQueue,
        dir: Directory,
    ):
        super().__init__()

        self.connection = connection
        self.dir = dir
        self.logging_queue = logging_queue

        self.name = self.name.replace("WorkerProcess", "Worker")

        self.start()

    def run(self):
        # Logging setup
        logging_qh = logging.handlers.QueueHandler(self.logging_queue)
        logging_root = logging.getLogger()
        logging_root.setLevel(logging.INFO)
        logging_root.addHandler(logging_qh)
        self.logger = logging.getLogger("log")

        # Main
        self.logger.info("Start")

        for task, args in iter(self.connection.recv, SENTINEL):
            try:
                self.logger.info(f"{'start':5} {task!s}")
                self.connection.send(WorkerResult(task, task(self.dir, **args)))
                self.logger.info(f"{'end':5} {task!s}")
            except Exception:
                self.logger.exception("Task error")
                self.connection.send(WorkerResult(task, SENTINEL))

        self.logger.info("Kill")
