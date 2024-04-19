import logging.config
from multiprocessing import JoinableQueue, Process, Queue
from threading import Thread
from typing import cast

from numpy.random import default_rng

from .argument_parser import parse_args
from .logging import create_logging_config_dict, logger_thread
from .path import Directory
from .precedence import task_precedence
from .task import TaskExecutor, task_manager
from .worker import file_thread, worker

# Parse arguments
args = parse_args()

# Create directory
dir = Directory(args.name)
dir.mkdir()

# Create random generators
seeds = (
    args.seeds
    if isinstance(args.seeds, list)
    else cast(
        list[int], default_rng(args.seed).integers(2**63, size=args.seeds).tolist()
    )
)
with dir.seeds_file.open("w") as f:
    for i, seed in enumerate(seeds):
        f.write(f"{i},{seed}\n")

# Create queues
task_queue: JoinableQueue = JoinableQueue()
done_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()
train_results_queue: Queue = Queue()
test_results_queue: Queue = Queue()

# Set up task precedence
to_do, succeed, precede = task_precedence(args)

# Populate task_queue
for task in to_do:
    task_queue.put(task)

# Create task executor
task_executor = TaskExecutor(
    args,
    dir,
    seeds,
    train_results_queue,
    test_results_queue,
)

# Start result file threads
train_result_thread = Thread(
    target=file_thread,
    args=(
        dir.train_results_file,
        train_results_queue,
    ),
)
test_result_thread = Thread(
    target=file_thread,
    args=(
        dir.test_results_file,
        test_results_queue,
    ),
)

train_result_thread.start()
test_result_thread.start()

# Start task manager
Process(target=task_manager, args=(succeed, precede, task_queue, done_queue)).start()

# Start workers
for i in range(args.jobs):
    Process(
        target=worker,
        args=(task_executor, task_queue, done_queue, logging_queue),
    ).start()

# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()

# Wait all tasks to be done
task_queue.join()

# Stop workers
for i in range(args.jobs):
    task_queue.put("STOP")

# Stop task manager
done_queue.put("STOP")

# Stop result file threads
train_results_queue.put("STOP")
test_results_queue.put("STOP")
train_result_thread.join()
test_result_thread.join()

# Stop logging thread
logging_queue.put("STOP")
logging_thread.join()
