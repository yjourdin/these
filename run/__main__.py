import logging.config
from collections import defaultdict
from multiprocessing import JoinableQueue, Process, Queue
from threading import Thread

from numpy.random import default_rng

from .argument_parser import parse_args
from .logging import create_logging_config_dict, logger_thread
from .path import Directory
from .task import (
    Task,
    TaskExecutor,
    task_A_test,
    task_A_train,
    task_D,
    task_manager,
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
seeds = (
    args.seeds
    if isinstance(args.seeds, list)
    else default_rng(args.seed).integers(2**63, size=args.seeds)
)

# Create queues
task_queue: JoinableQueue = JoinableQueue()
done_queue: JoinableQueue = JoinableQueue()
logging_queue: Queue = Queue()
train_results_queue: Queue = Queue()
test_results_queue: Queue = Queue()

# Set up task precedence
succeed: defaultdict[Task, list[Task]] = defaultdict(list)
precede: defaultdict[Task, list[Task]] = defaultdict(list)

for i in range(len(seeds)):
    for m in args.M:
        for n_te in args.N_te:
            t_A_test = task_A_test(i, n_te, m)
            task_queue.put(t_A_test)  # Put task in task queue
    for n_tr in args.N_tr:
        t_A_train = task_A_train(i, n_tr, m)
        task_queue.put(t_A_train)  # Put task in task queue
        for Mo in args.Mo:
            for ko in args.Ko:
                t_Mo = task_Mo(i, m, Mo, ko)
                task_queue.put(t_Mo)  # Put task in task queue
                for n_bc in args.N_bc:
                    for e in args.error:
                        t_D = task_D(i, n_tr, m, Mo, ko, n_bc, e)
                        succeed[t_A_train] += [t_D]
                        succeed[t_Mo] += [t_D]
                        precede[t_D] += [t_A_train, t_Mo]
                        for Me in args.Me:
                            for ke in args.Ke:
                                ts = []
                                for method in args.method:
                                    match method:
                                        case "SA":
                                            for config in range(len(args.config) or 1):
                                                t_Me = task_SA(
                                                    i,
                                                    n_tr,
                                                    m,
                                                    Mo,
                                                    ko,
                                                    n_bc,
                                                    e,
                                                    Me,
                                                    ke,
                                                    config,
                                                )
                                                t_test = task_test(
                                                    i,
                                                    n_tr,
                                                    n_te,
                                                    m,
                                                    Mo,
                                                    ko,
                                                    n_bc,
                                                    e,
                                                    Me,
                                                    ke,
                                                    method,
                                                    config,
                                                )
                                                ts.append((t_Me, t_test))
                                        case "MIP" if Me == "SRMP":
                                            t_Me = task_MIP(
                                                i, n_tr, m, Mo, ko, n_bc, e, ke
                                            )
                                            t_test = task_test(
                                                i,
                                                n_tr,
                                                n_te,
                                                m,
                                                Mo,
                                                ko,
                                                n_bc,
                                                e,
                                                Me,
                                                ke,
                                                method,
                                            )
                                            ts.append((t_Me, t_test))
                                        case _:
                                            break
                                    for n_te in args.N_te:
                                        for t_Me, t_test in ts:
                                            succeed[t_D] += [t_Me]
                                            precede[t_Me] += [t_D]
                                            succeed[t_Me] += [t_test]
                                            succeed[t_A_test] += [t_test]
                                            precede[t_test] += [t_A_test, t_Me]

# Create task executor
task_executor = TaskExecutor(
    args,
    dir,
    seeds,
    train_results_queue,
    test_results_queue,
    args.config or {},
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
