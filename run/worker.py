import logging
import logging.handlers
from multiprocessing import JoinableQueue, Queue

from .task import Task, TaskExecutor


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
            done_queue.put(task)
            done_queue.join()
            logger.info(task_executor.name(task) + " done")
            task_queue.task_done()
        except Exception as e:
            logger.error(e, exc_info=True)
            break

    logger.info("Kill")
    task_queue.task_done()


def file_thread(file, q):
    with file.open("a") as f:
        for result in iter(q.get, "STOP"):
            f.write(result)
            f.flush()
