import logging.config
from multiprocessing import Event, JoinableQueue, Pipe, Process, Queue
from multiprocessing.connection import Connection
from threading import Thread

from ..constants import SENTINEL
from .argument_parser import parse_args
from .directory import Directory
from .fieldnames import ConfigFieldnames
from .logging import create_logging_config_dict
from .precedence import task_precedence
from .thread.logger import logger_thread
from .thread.stop import stopping_thread
from .thread.task_manager import task_manager
from .worker import worker

# Create stop event
stop_event = Event()

# Parse arguments
args = parse_args()


# Create directory
dir = Directory(args.dir, args.name)
dir.mkdir()


# Start file threads
for thread in dir.csv_files.threads.values():
    thread.start()


# Populate args
args.complete()


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


# Create queues
task_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()


# Set up task precedence
to_do, succeed, precede, follow_up = task_precedence(args)


# Start stoppig thread
stop_thread = Thread(target=stopping_thread, args=(stop_event, dir.run, task_queue))
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
            task_queue,
            worker_connection,
            logging_queue,
            stop_event,
            args.stop_error,
        ),
    )
    connections.append(main_connection)
    worker_process.start()
    workers.append(worker_process)


# Start task manager thread
task_manager_thread = Thread(
    target=task_manager,
    args=(succeed, precede, follow_up, task_queue, connections, stop_event),
)
task_manager_thread.start()


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
task_manager_thread.join()


# Stop threads
logging_queue.put(SENTINEL)
for queue in dir.csv_files.queues.values():
    queue.put(SENTINEL)

logging_thread.join()
for thread in dir.csv_files.threads.values():
    thread.join()
