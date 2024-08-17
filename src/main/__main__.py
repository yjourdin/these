import csv
import logging.config
from multiprocessing import Event, JoinableQueue, Process, Queue
from threading import Thread

from .argument_parser import parse_args
from .directory import FIELDNAMES, Directory
from .logging import create_logging_config_dict, logger_thread
from .precedence import task_precedence
from .worker import csv_file_thread, event_thread, task_manager, worker

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
    writer = csv.DictWriter(f, FIELDNAMES[dir.configs.stem], dialect="unix")
    for config in args.config:
        writer.writerow(
            {
                "Id": config.id,
                "Method": config.method,
                "Config": {k: v for k, v in config.to_dict().items() if k != "id"},
            }
        )


# Create queues
task_queue: JoinableQueue = JoinableQueue()
done_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()
seeds_queue: Queue = Queue()
train_results_queue: Queue = Queue()
test_results_queue: Queue = Queue()


# Set up task precedence
to_do, succeed, precede = task_precedence(args)


# Populate task_queue
for task in to_do:
    task_queue.put(task)


# Start file threads
seeds_thread = Thread(
    target=csv_file_thread,
    args=(
        dir.seeds,
        seeds_queue,
    ),
)
train_results_thread = Thread(
    target=csv_file_thread,
    args=(
        dir.train_results,
        train_results_queue,
    ),
)
test_results_thread = Thread(
    target=csv_file_thread,
    args=(
        dir.test_results,
        test_results_queue,
    ),
)

seeds_thread.start()
train_results_thread.start()
test_results_thread.start()


# Start task manager
task_manager_thread = Thread(
    target=task_manager, args=(succeed, precede, task_queue, done_queue)
)
task_manager_thread.start()


# Start workers
workers: list[Process] = []
for i in range(args.jobs):
    worker_process = Process(
        target=worker,
        args=(
            task_queue,
            done_queue,
            logging_queue,
            dir,
            {
                "seeds": seeds_queue,
                "train": train_results_queue,
                "test": test_results_queue,
            },
        ),
    )
    worker_process.start()
    workers.append(worker_process)


# Start logging thread
logging.config.dictConfig(create_logging_config_dict(dir.log))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Start stop thread
stop_event = Event()
stop_thread = Thread(
    target=event_thread,
    args=(
        stop_event,
        dir.run,
        task_queue,
    ),
)
stop_thread.start()


# Wait all tasks to be done
task_queue.join()


# Stop stopping thread
stop_event.set()
stop_thread.join()


# Stop workers
for _ in range(args.jobs):
    task_queue.put("STOP")
for worker_process in workers:
    if stop_event.is_set():
        worker_process.terminate()
        task_queue.task_done()
    worker_process.join()
task_queue.join()
task_queue.close()
task_queue.cancel_join_thread()


# Stop task manager
done_queue.put("STOP")
task_manager_thread.join()
if stop_event.is_set():
    while not done_queue.empty():
        done_queue.get()
        done_queue.task_done()
    try:
        while True:
            done_queue.task_done()
    except ValueError:
        ...
done_queue.join()
done_queue.close()
done_queue.cancel_join_thread()


# Stop threads
seeds_queue.put("STOP")
train_results_queue.put("STOP")
test_results_queue.put("STOP")
logging_queue.put("STOP")

seeds_thread.join()
train_results_thread.join()
test_results_thread.join()
logging_thread.join()

seeds_queue.close()
train_results_queue.close()
test_results_queue.close()
logging_queue.close()

seeds_queue.cancel_join_thread()
train_results_queue.cancel_join_thread()
test_results_queue.cancel_join_thread()
logging_queue.cancel_join_thread()
