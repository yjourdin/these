import logging
import logging.handlers
from multiprocessing import JoinableQueue, Queue

from .task import TaskExecutor, Task


def worker(
    task_executor: TaskExecutor,
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
    logging_queue: Queue,
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)

    logger = logging.getLogger("log")
    for task in iter(task_queue.get, "STOP"):
        try:
            logger.info(task_executor.name(task) + " running...")
            task_executor.execute(task)
            task_queue.task_done()
            done_queue.put(task)
            logger.info(task_executor.name(task) + " done")
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.info("Kill")
            break

    task_queue.task_done()


def file_thread(file, q):
    with file.open("a") as f:
        for result in iter(q.get, "STOP"):
            f.write(result)
