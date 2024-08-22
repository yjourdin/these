from collections import defaultdict
from multiprocessing import JoinableQueue

from ...constants import SENTINEL
from ..task import Task


def task_manager(
    succeed: defaultdict[Task, list[Task]],
    precede: defaultdict[Task, list[Task]],
    task_queue: "JoinableQueue[Task]",
    done_queue: "JoinableQueue[Task]",
):
    task_set = set()
    done_set = set()
    for task in iter(done_queue.get, SENTINEL):
        done_set.add(task)
        for next_task in succeed[task]:
            if next_task not in task_set:
                if all(t in done_set for t in precede[next_task]):
                    task_queue.put(next_task)
                    task_set.add(next_task)
        done_queue.task_done()
    done_queue.task_done()
