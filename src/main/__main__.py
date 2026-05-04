import logging.config
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection
from threading import Thread

from src.constants import SENTINEL

from .args import ARGS
from .arguments import ExperimentEnum
from .connection import WorkerPipe
from .directory import Directory
from .experiments.elicitation.main import main as main_elicitation
from .experiments.group_decision.main import main as main_group_decision
from .init_directory import DIR
from .logging import LoggingQueue, create_logging_config_dict
from .threads.csv_file import CSVFileThread
from .threads.logger import LoggerThread
from .threads.stop import STOP, StopThread
from .threads.task_manager import TASK_QUEUE, TaskManager
from .worker import WorkerProcess

# Set main function
directory_class = Directory
match ARGS.experiment:
    case ExperimentEnum.ELICITATION:
        MAIN = main_elicitation
    case ExperimentEnum.GROUP_DECISION:
        MAIN = main_group_decision  # pyright: ignore[reportConstantRedefinition]


# Start file threads

csv_threads: list[Thread] = []
for csv_file in DIR.itercsv():
    thread = CSVFileThread(csv_file)
    csv_threads.append(thread)


# Create logging queue

logging_queue: LoggingQueue = Queue()


# Start logging thread

logging.config.dictConfig(create_logging_config_dict(DIR))
logging_thread = LoggerThread(logging_queue)


# Start worker processes

workers: list[Process] = []
connections: list[Connection] = []
for i in range(ARGS.jobs):
    worker_connection, manager_connection = WorkerPipe()
    worker_process = WorkerProcess(worker_connection, logging_queue, DIR)
    connections.append(manager_connection)
    workers.append(worker_process)


# Start worker manager thread

task_manager_thread = TaskManager(connections)


# Start stop thread

stop_thread = StopThread(DIR.run)


# Start main thread
main_thread = Thread(target=MAIN, args=(ARGS,))
main_thread.start()


# Wait for main thread to finish or stopping condition
while main_thread.is_alive() and not STOP.is_set():
    main_thread.join(1)


# Send stop signal
STOP.set()


# Join main thread
main_thread.join()


# Join stop thread
stop_thread.join()


# Join task manager thread
task_manager_thread.join()


# Join workers
for worker_process in workers:
    if worker_process.is_alive():
        worker_process.terminate()
    worker_process.join()


# Join task queue
TASK_QUEUE.join()


# Join logging thread

logging_queue.put(SENTINEL)
logging_thread.join()


# Close directory
DIR.close()
for csv_thread in csv_threads:
    csv_thread.join()
