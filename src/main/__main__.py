import logging.config
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from threading import Thread

from ..constants import SENTINEL
from .argument_parser import parse_args
from .directory import Directory
from .fieldnames import ConfigFieldnames
from .logging import create_logging_config_dict
from .precedence import task_precedence
from .threads.logger import logger_thread
from .threads.stop import stopping_thread
from .threads.task_manager import task_manager
from .worker import worker

# Parse arguments
args = parse_args()


# Populate args
args.complete()


# Initialise directory
dir = Directory(args.dir, args.name)


if not args.extend:
    # Create Directory
    dir.mkdir()

    # Write arguments
    with dir.args.open("w") as f:
        f.write(args.to_json())

    # Write configs
    for config in args.config:
        dir.csv_files["configs"].queue.put(
            {
                ConfigFieldnames.Id: config.id,
                ConfigFieldnames.Method: config.method,
                ConfigFieldnames.Config: {
                    k: v for k, v in config.to_dict().items() if k != "id"
                },
            }
        )


# Create run file
dir.run.touch()


# Start file threads
for thread in dir.csv_files.threads.values():
    thread.start()


# Create logging queue
logging_queue: Queue = Queue()


# Set up task precedence
start, succeed, precede, priority_succeed = task_precedence(args)


# Start stopping thread
stop_connection, task_manager_connection = Pipe()
stop_thread = Thread(target=stopping_thread, args=(dir.run, stop_connection))
stop_thread.start()


# Start worker processes
workers: list[Process] = []
connections: list[Connection] = []
for i in range(args.jobs):
    worker_connection, main_connection = Pipe()
    worker_process = Process(
        target=worker,
        args=(
            dir,
            worker_connection,
            logging_queue,
            args.stop_error,
        ),
    )
    connections.append(main_connection)
    worker_process.start()
    workers.append(worker_process)


# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Start task manager thread
task_manager_thread = Thread(
    target=task_manager,
    args=(
        succeed,
        precede,
        priority_succeed,
        start,
        connections,
        task_manager_connection,
    ),
)
task_manager_thread.start()


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
for queue in dir.csv_files.queues.values():
    queue.put(SENTINEL)

logging_thread.join()
for thread in dir.csv_files.threads.values():
    thread.join()
