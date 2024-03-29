import logging.config
from collections import defaultdict
from multiprocessing import JoinableQueue, Manager, Process, Queue
from threading import Thread

from numpy.random import default_rng

from .argument_parser import parse_args
from .logging import create_logging_config_dict, logger_thread
from .path import Directory
from .task import (
    Task,
    TaskManager,
    task_A_test,
    task_A_train,
    task_D,
    task_MIP,
    task_Mo,
    task_SA,
    task_test,
)
from .worker import file_thread, worker

# Parse arguments
args = parse_args()

# Create directory
dir = Directory(args.name)
dir.mkdir()

# Create random generators
rngs = default_rng(args.seed).spawn(args.N_exp)

# Create queues
task_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()
train_results_queue: Queue = Queue()
test_results_queue: Queue = Queue()

# Create TaskManager
succeed: defaultdict[Task, list[Task]] = defaultdict(list)
precede: defaultdict[Task, list[Task]] = defaultdict(list)

for i in range(args.N_exp):
    for m in args.M:
        t_A_train = task_A_train(i, m)
        t_A_test = task_A_test(i, m)
        task_queue.put(t_A_train)  # Put task in task queue
        task_queue.put(t_A_test)  # Put task in task queue
        for Mo in args.Mo:
            for ko in args.Ko:
                t_Mo = task_Mo(i, m, Mo, ko)
                task_queue.put(t_Mo)  # Put task in task queue
                for n_bc in args.N_bc:
                    for e in args.error:
                        t_D = task_D(i, m, Mo, ko, n_bc, e)
                        precede[t_D] += [t_A_train, t_Mo]
                        succeed[t_A_train] += [t_D]
                        succeed[t_Mo] += [t_D]
                        for Me in args.Me:
                            for ke in args.Ke:
                                for method in args.method:
                                    match method:
                                        case "SA":
                                            t_Me = task_SA(
                                                i, m, Mo, ko, n_bc, e, Me, ke
                                            )
                                        case "MIP" if Me == "SRMP":
                                            t_Me = task_MIP(i, m, Mo, ko, n_bc, e, ke)
                                        case _:
                                            break
                                    t_test = task_test(i, m, Mo, ko, n_bc, e, Me, ke)
                                    precede[t_Me] += [t_D]
                                    succeed[t_D] += [t_Me]
                                    precede[t_test] += [t_A_test, t_Me]
                                    succeed[t_Me] += [t_test]
                                    succeed[t_A_test] += [t_test]

task_manager = TaskManager(
    args, succeed, precede, dir, rngs, train_results_queue, test_results_queue
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


with Manager() as manager:
    done_dict = manager.dict()
    put_dict = manager.dict()

    # Start workers
    for i in range(args.jobs):
        Process(
            target=worker,
            args=(task_manager, task_queue, put_dict, done_dict, logging_queue),
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

    # Stop result file threads
    train_results_queue.put("STOP")
    test_results_queue.put("STOP")
    train_result_thread.join()
    test_result_thread.join()

    # Stop logging thread
    logging_queue.put("STOP")
    logging_thread.join()
