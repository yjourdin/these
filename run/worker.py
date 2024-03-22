import logging
import logging.handlers
from multiprocessing import JoinableQueue, Queue
from multiprocessing.managers import DictProxy

from .task import Task, TaskManager


def worker(
    task_manager: TaskManager,
    task_queue: JoinableQueue,
    put_dict: "DictProxy[Task, bool]",
    done_dict: "DictProxy[Task, bool]",
    logging_queue: Queue,
):
    logging_qh = logging.handlers.QueueHandler(logging_queue)
    logging_root = logging.getLogger()
    logging_root.setLevel(logging.INFO)
    logging_root.addHandler(logging_qh)

    for task in iter(task_queue.get, "STOP"):
        try:
            task_manager.execute(task)
            done_dict[task] = True
            for t in task_manager.next_tasks(task, done_dict):
                if not put_dict.get(t, False):
                    task_queue.put(t)
                    put_dict[t] = True
            task_queue.task_done()
        except Exception as e:
            logger = logging.getLogger("log")
            logger.error(e)
            logger.info("Kill")
            break

    task_queue.task_done()


def file_thread(file, q):
    with file.open("a") as f:
        for result in iter(q.get, "STOP"):
            f.write(result)
