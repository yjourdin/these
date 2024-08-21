import csv
import logging.config
from multiprocessing import Event, JoinableQueue, Process, Queue
from threading import Thread

from .argument_parser import parse_args
from .csv_file import CSVFile, CSVFiles
from .directory import Directory
from .fieldnames import ConfigFieldnames
from .logging import create_logging_config_dict
from .precedence import task_precedence
from .worker import SENTINEL, logger_thread, stopping_thread, task_manager, worker

# Create stop event
stop_event = Event()

# Parse arguments
args = parse_args()


# Create directory
dir = Directory(args.dir, args.name)
dir.mkdir()


# Populate args
args.complete()


# Write arguments
with dir.args.open("w") as f:
    f.write(args.to_json())


# Write configs
with dir.configs.open("a", newline="") as f:
    writer = csv.DictWriter(f, ConfigFieldnames, dialect="unix")
    for config in args.config:
        writer.writerow(
            {
                ConfigFieldnames.Id: config.id,
                ConfigFieldnames.Method: config.method,
                ConfigFieldnames.Config: {
                    k: v for k, v in config.to_dict().items() if k != "id"
                },
            }
        )


# Create queues
task_queue: JoinableQueue = JoinableQueue()
done_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()


# Set up task precedence
to_do, succeed, precede = task_precedence(args)


# Start stoppig thread
stop_thread = Thread(target=stopping_thread, args=(stop_event, dir.run, task_queue))
stop_thread.start()


# Start file threads

files: CSVFiles = CSVFiles(
    {
        "seeds": CSVFile(dir.seeds),
        "D_size": CSVFile(dir.D_size),
        "train": CSVFile(dir.train_results),
        "test": CSVFile(dir.test_results),
    }
)
for thread in files.threads.values():
    thread.start()


# Start task manager thread
task_manager_thread = Thread(
    target=task_manager, args=(succeed, precede, task_queue, done_queue)
)
task_manager_thread.start()


# Start worker processes
workers: list[Process] = []
for i in range(args.jobs):
    worker_process = Process(
        target=worker,
        args=(task_queue, done_queue, logging_queue, stop_event, dir, files.queues),
    )
    worker_process.start()
    workers.append(worker_process)


# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Populate task_queue
for task in to_do:
    task_queue.put(task)


# Wait all tasks to be done
task_queue.join()


# Stop stopping thread
NORMAL_EXIT = not stop_event.is_set()
if stop_thread.is_alive():
    stop_event.set()
    stop_thread.join()


# Stop workers
if NORMAL_EXIT:
    for _ in range(args.jobs):
        task_queue.put(SENTINEL)
    task_queue.join()
else:
    for worker_process in workers:
        worker_process.terminate()


# Stop task manager
done_queue.put(SENTINEL)
done_queue.join()


# Stop threads
logging_queue.put(SENTINEL)
for queue in files.queues.values():
    queue.put(SENTINEL)

logging_thread.join()
for thread in files.threads.values():
    thread.join()
