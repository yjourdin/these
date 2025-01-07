import csv
from concurrent.futures import Future, ThreadPoolExecutor
from shutil import copy
from time import process_time
from typing import Any

from .....preference_structure.io import from_csv, to_csv
from .....preference_structure.utils import refused_preferences
from .....utils import raise_exception, raise_exceptions
from ....threads.task import task_thread
from ....threads.worker_manager import TaskQueue
from ..directory import DirectoryGroupDecision
from ..task import AcceptTask, CollectiveTask, PreferencePathTask


def collective_thread(
    args: dict[str, Any],
    task_queue: TaskQueue,
    precede_futures: list[Future],
    dir: DirectoryGroupDecision,
):
    raise_exceptions(precede_futures)

    with ThreadPoolExecutor() as thread_pool:
        it = 0

        task_Mc = CollectiveTask(
            args["m"],
            args["n_tr"],
            args["Atr_id"],
            args["ko"],
            args["Mo_id"],
            args["group_size"],
            args["gen"],
            args["Mi_id"],
            args["n_bc"],
            args["same_alt"],
            args["D_id"],
            args["config"],
            args["Mc_id"],
            it,
        )

        with task_Mc.C_file(dir).open("w", newline="") as f:
            C_writer = csv.writer(f, dialect="unix")
            C_writer.writerows([[0]] * args["group_size"])

        
        for dm_id in range(args["group_size"]):
            task_Mc.R_dir(dir, dm_id).mkdir(exist_ok=True)
        
        start_time = process_time()
        compromise_found = False
        while not compromise_found:
            future_Mc = thread_pool.submit(
                task_thread,
                task_Mc,
                {"seed": args["seeds"].Mc[args["Mc_id"]]},
                task_queue,
                [],
                dir,
            )

            raise_exception(future_Mc)
            
            tasks_P: dict[int, PreferencePathTask] = {}
            futures_P: dict[int, Future] = {}
            for dm_id in range(args["group_size"]):
                tasks_P[dm_id] = PreferencePathTask(
                    args["m"],
                    args["n_tr"],
                    args["Atr_id"],
                    args["ko"],
                    args["Mo_id"],
                    args["group_size"],
                    args["gen"],
                    args["Mi_id"],
                    dm_id,
                    args["n_bc"],
                    args["same_alt"],
                    args["D_id"],
                    args["config"],
                    args["Mc_id"],
                    it,
                )
                futures_P[dm_id] = thread_pool.submit(
                    task_thread, tasks_P[dm_id], {}, task_queue, [future_Mc], dir
                )

            raise_exceptions(futures_P.values())

            t = 1
            dms = range(args["group_size"])
            dms = [dm_id for dm_id in dms if tasks_P[dm_id].P_file(dir, t).exists()]

            dms_refusing: list[int] = []
            # for dm_id in range(args["group_size"]):
            #     copy(
            #         tasks_P[dm_id].D_file(dir, dm_id, it),
            #         tasks_P[dm_id].D_file(dir, dm_id, it + 1),
            #     )

            while dms:
                tasks_accept: dict[int, AcceptTask] = {}
                futures_accept: dict[int, Future] = {}

                for dm_id in dms:
                    # print(repr(args["accept"]))
                    tasks_accept[dm_id] = AcceptTask(
                        args["m"],
                        args["n_tr"],
                        args["Atr_id"],
                        args["ko"],
                        args["Mo_id"],
                        args["group_size"],
                        args["gen"],
                        args["Mi_id"],
                        dm_id,
                        args["n_bc"],
                        args["same_alt"],
                        args["D_id"],
                        args["config"],
                        args["Mc_id"],
                        it,
                        args["accept"],
                        t,
                    )
                    futures_accept[dm_id] = thread_pool.submit(
                        task_thread,
                        tasks_accept[dm_id],
                        {},
                        task_queue,
                        [futures_P[dm_id]],
                        dir,
                    )

                dms_refusing = [
                    dm_id
                    for dm_id, future in futures_accept.items()
                    if not future.result()
                ]

                if dms_refusing:
                    break

                t += 1
                dms = [dm_id for dm_id in dms if tasks_P[dm_id].P_file(dir, t).exists()]

            it += 1
            new_task_Mc = CollectiveTask(
                args["m"],
                args["n_tr"],
                args["Atr_id"],
                args["ko"],
                args["Mo_id"],
                args["group_size"],
                args["gen"],
                args["Mi_id"],
                args["n_bc"],
                args["same_alt"],
                args["D_id"],
                args["config"],
                args["Mc_id"],
                it,
            )

            changes: list[int] = []
            with task_Mc.C_file(dir).open("r", newline="") as f:
                C_reader = csv.reader(f, dialect="unix")
                for row in C_reader:
                    changes.append(int(row[0]))

            with new_task_Mc.C_file(dir).open("w", newline="") as f:
                C_writer = csv.writer(f, dialect="unix")

                for dm_id in range(args["group_size"]):
                    temp = 1
                    while (temp < t) and tasks_P[dm_id].P_file(dir, temp).exists():
                        temp += 1
                    accepted_t = temp - 1

                    copy(
                        tasks_P[dm_id].P_file(dir, accepted_t),
                        new_task_Mc.D_file(dir, dm_id),
                    )

                    with tasks_P[dm_id].P_file(dir, 0).open("r") as P0f:
                        P0 = from_csv(P0f)
                        with tasks_P[dm_id].P_file(dir, accepted_t).open("r") as P1f:
                            P1 = from_csv(P1f)
                            C_writer.writerow([changes[dm_id] + len(P1 - P0)])

                            if dm_id in dms_refusing:
                                with (
                                    tasks_P[dm_id]
                                    .P_file(dir, accepted_t + 1)
                                    .open("r") as P2f
                                ):
                                    P2 = from_csv(P2f)

                                    with new_task_Mc.R_file(dir, dm_id).open(
                                        "w"
                                    ) as R_file:
                                        to_csv(refused_preferences(P1, P2), R_file)

            if dms_refusing:
                task_Mc = new_task_Mc
                time = process_time() - start_time
                if time > 3600:
                    break
            else:
                compromise_found = True
