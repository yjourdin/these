import logging.config
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue, set_start_method
from multiprocessing.connection import Connection
from queue import LifoQueue
from threading import Thread
from typing import Any

from ..constants import SENTINEL
from .argument_parser import parse_args
from .arguments import ExperimentEnum
from .connection import StopPipe, WorkerPipe
from .directory import Directory
from .experiments.elicitation.directory import DirectoryElicitation
from .experiments.elicitation.main import main as main_elicitation
from .experiments.group_decision.directory import DirectoryGroupDecision
from .experiments.group_decision.main import main as main_group_decision
from .logging import create_logging_config_dict
from .threads.csv_file import csv_file_thread
from .threads.logger import logger_thread
from .threads.stop import stopping_thread
from .threads.worker_manager import TaskQueue, worker_manager
from .worker import worker

# Set process start method
set_start_method("forkserver")


# Parse arguments
args = parse_args()


# Set experiment
directory_class = Directory
match args.experiment:
    case ExperimentEnum.ELICITATION:
        directory_class = DirectoryElicitation
        main = main_elicitation
    case ExperimentEnum.GROUP_DECISION:
        directory_class = DirectoryGroupDecision
        main = main_group_decision


# Initialise directory
dir = directory_class(args.name, args.dir)


if not args.extend:
    # Create Directory
    dir.mkdir()

    # Write arguments
    with dir.args.open("w") as f:
        f.write(args.to_json())


# Create logging queue
logging_queue: "Queue[Any]" = Queue()


# Start worker processes
workers: list[Process] = []
connections: list[Connection] = []
for i in range(args.jobs):
    worker_connection, manager_connection = WorkerPipe()
    worker_process = Process(
        target=worker, args=(worker_connection, logging_queue, dir)
    )
    connections.append(manager_connection)
    worker_process.start()
    workers.append(worker_process)


# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Start file threads
csv_threads: list[Thread] = []
for csv_file in dir.itercsv():
    thread = Thread(target=csv_file_thread, args=(csv_file,))
    csv_threads.append(thread)
    thread.start()


# Create task queue
task_queue: TaskQueue = LifoQueue()


# Create run file
dir.run.touch()


# Start stopping thread
stop_connection, manager_connection = StopPipe()
stop_thread = Thread(
    target=stopping_thread, args=(dir.run, stop_connection, task_queue)
)
stop_thread.start()


with ThreadPoolExecutor(300) as thread_pool:
    # Start worker manager thread
    task_manager_thread = Thread(
        target=worker_manager,
        args=(
            connections,
            task_queue,
            manager_connection,
            thread_pool,
            args.stop_error,
        ),
    )
    task_manager_thread.start()

    # Main
    try:
        main(args, dir, thread_pool, task_queue)  # type: ignore
    except Exception:
        traceback.print_exc()

    # Send stop signal
    task_queue.put(SENTINEL)

    # Join task manager thread
    task_manager_thread.join()

    # Join stopping thread
    stop_thread.join()

    # Join workers
    for worker_process in workers:
        if worker_process.is_alive():
            worker_process.terminate()
        worker_process.join()

    # Join threads
    logging_queue.put(SENTINEL)
    logging_thread.join()

    for csv_file in dir.itercsv():
        csv_file.queue.put(SENTINEL)

    for csv_thread in csv_threads:
        csv_thread.join()
