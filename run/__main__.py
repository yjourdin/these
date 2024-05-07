import csv
import logging.config
from multiprocessing import JoinableQueue, Process, Queue
from threading import Thread

from .argument_parser import parse_args
from .logging import create_logging_config_dict, logger_thread
from .path import Directory
from .precedence import task_precedence
from .seed import create_seeds
from .task import TaskExecutor, task_manager
from .worker import csv_file_thread, worker

# Parse arguments
args = parse_args()

# Create directory
dir = Directory(args.name)
dir.mkdir()

# Create random seeds
seeds = create_seeds(args)
with dir.seeds_file.open("a", newline='') as f:
    writer = csv.writer(f, "unix")
    for i, seed in enumerate(seeds["A_train"]):
        writer.writerow(["A_train", i, seed])
    for i, seed in enumerate(seeds["A_test"]):
        writer.writerow(["A_test", i, seed])
    for i, seed in enumerate(seeds["Mo"]):
        writer.writerow(["Mo", i, seed])

# Write configs
with dir.configs_file.open("a", newline='') as f:
    writer = csv.writer(f, "unix")
    for method, configs in args.config.items():
        for id, config in configs.items():
            writer.writerow([method, id, config])


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
    target=csv_file_thread,
    args=(
        dir.train_results_file,
        train_results_queue,
    ),
)
test_result_thread = Thread(
    target=csv_file_thread,
    args=(
        dir.test_results_file,
        test_results_queue,
    ),
)
train_result_thread.start()
test_result_thread.start()

# Start task manager
task_manager_process = Process(
    target=task_manager, args=(succeed, precede, task_queue, done_queue)
)
task_manager_process.start()

# Start workers
workers: list[Process] = []
for i in range(args.jobs):
    worker_process = Process(
        target=worker,
        args=(task_executor, task_queue, done_queue, logging_queue),
    )
    worker_process.start()
    workers.append(worker_process)

# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Wait all tasks to be done
task_queue.join()


# Stop workers
for i in range(args.jobs):
    task_queue.put("STOP")
task_queue.join()
for i in range(args.jobs):
    workers[i].join()

# Stop task manager
done_queue.put("STOP")
done_queue.join()
task_manager_process.join()

# Stop result file threads
train_results_queue.put("STOP")
test_results_queue.put("STOP")
train_result_thread.join()
test_result_thread.join()

# Stop logging thread
logging_queue.put("STOP")
logging_thread.join()
