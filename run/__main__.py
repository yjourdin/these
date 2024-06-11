import csv
import logging.config
from dataclasses import asdict
from multiprocessing import JoinableQueue, Process, Queue
from threading import Thread

from numpy.random import default_rng

from .argument_parser import parse_args
from .config import create_config
from .logging import create_logging_config_dict, logger_thread
from .path import FIELDNAMES, Directory
from .precedence import task_precedence
from .seed import seeds
from .worker import csv_file_thread, task_manager, worker

# Parse arguments
args = parse_args()


# Create directory
dir = Directory(args.name)
dir.mkdir()


# Create random seeds
rng = default_rng(args.seed)

args.seeds.A_train = args.seeds.A_train + seeds(
    rng, args.nb_A_tr - len(args.seeds.A_train)
)
args.seeds.Mo = args.seeds.Mo + seeds(
    rng, (args.nb_Mo or args.nb_A_tr) - len(args.seeds.Mo)
)
args.seeds.A_test = args.seeds.A_test or seeds(
    rng, (args.nb_A_te or args.nb_Mo or args.nb_A_tr) - len(args.seeds.A_test)
)

# Create missing configs
for method in args.method:
    if not any(config.method == method for config in args.config):
        args.config.append(create_config(method=method))


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
                "Config": {k: v for k, v in asdict(config).items() if k != "id"},
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
task_manager_process = Process(
    target=task_manager, args=(succeed, precede, task_queue, done_queue)
)
task_manager_process.start()


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
            args.seeds,
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
logging.config.dictConfig(create_logging_config_dict(dir))
logging_thread = Thread(target=logger_thread, args=(logging_queue,))
logging_thread.start()


# Wait all tasks to be done
task_queue.join()


# Stop workers
for _ in range(args.jobs):
    task_queue.put("STOP")
task_queue.join()
for worker_process in workers:
    worker_process.join()


# Stop task manager
done_queue.put("STOP")
done_queue.join()
task_manager_process.join()


# Stop file threads
seeds_queue.put("STOP")
train_results_queue.put("STOP")
test_results_queue.put("STOP")

seeds_thread.join()
train_results_thread.join()
test_results_thread.join()


# Stop logging thread
logging_queue.put("STOP")
logging_thread.join()
