import csv
from concurrent.futures import Future, ThreadPoolExecutor
from shutil import copy
from typing import Any

from .....constants import DEFAULT_MAX_TIME
from .....preference_structure.io import from_csv, to_csv
from ....task import FutureTaskException, raise_exception, raise_exceptions
from ....threads.task import task_thread
from ....threads.worker_manager import TaskQueue
from ..directory import DirectoryGroupDecision
from ..fieldnames import CompromiseFieldnames
from ..task import AcceptTask, CleanTask, CollectiveTask, PreferencePathTask


def collective_thread(
    args: dict[str, Any],
    task_queue: TaskQueue,
    precede_futures: list[Future],
    dir: DirectoryGroupDecision,
    max_time: int = DEFAULT_MAX_TIME,
):
    raise_exceptions(precede_futures)

    with ThreadPoolExecutor() as thread_pool:
        it = 0
        changes: list[int] = [0] * args["group_size"]

        task_Mc = CollectiveTask(
            args["m"],
            args["n_tr"],
            args["Atr_id"],
            args["ko"],
            args["Mo_id"],
            args["group_size"],
            args["group"],
            args["Mi_id"],
            args["n_bc"],
            args["same_alt"],
            args["D_id"],
            args["config"],
            args["Mc_id"],
            args["P_id"],
            it,
        )

        with task_Mc.C_file(dir).open("w", newline="") as f:
            C_writer = csv.writer(f, dialect="unix")
            C_writer.writerows([[0]] * args["group_size"])

        # for dm_id in range(args["group_size"]):
        #     task_Mc.R_dir(dir, dm_id).mkdir(exist_ok=True)

        time_left = max_time
        compromise_found = False
        while (not compromise_found) and (time_left >= 1):
            future_Mc = thread_pool.submit(
                task_thread,
                task_Mc,
                {"seed": args["seeds"].Mc[args["Mc_id"]], "max_time": time_left},
                task_queue,
                [],
                dir,
            )

            if raise_exception(future_Mc):
                result, time = future_Mc.result()
            

            time_left -= time
            if time_left < 1:
                break

            if result:
                tasks_P: dict[int, PreferencePathTask] = {}
                futures_P: dict[int, FutureTaskException] = {}
                for dm_id in range(args["group_size"]):
                    tasks_P[dm_id] = PreferencePathTask(
                        args["m"],
                        args["n_tr"],
                        args["Atr_id"],
                        args["ko"],
                        args["Mo_id"],
                        args["group_size"],
                        args["group"],
                        args["Mi_id"],
                        dm_id,
                        args["n_bc"],
                        args["same_alt"],
                        args["D_id"],
                        args["config"],
                        args["Mc_id"],
                        args["P_id"],
                        it,
                    )
                    futures_P[dm_id] = thread_pool.submit(
                        task_thread,
                        tasks_P[dm_id],
                        {
                            "seed": args["seeds"].P[args["P_id"]],
                            "max_time": time_left,
                        },
                        task_queue,
                        [future_Mc],
                        dir,
                    )

                futures_P_values = futures_P.values()
                if raise_exceptions(futures_P_values):
                    time_left -= max(
                        future.result().time for future in futures_P_values
                    )
                if time_left < 1:
                    break

                t = 1
                dms = range(args["group_size"])
                dms = [dm_id for dm_id in dms if tasks_P[dm_id].P_file(dir, t).exists()]

                dms_refusing: list[int] = []

                while dms:
                    futures_accept: dict[int, FutureTaskException] = {}

                    for dm_id in dms:
                        tasks_accept = AcceptTask(
                            args["m"],
                            args["n_tr"],
                            args["Atr_id"],
                            args["ko"],
                            args["Mo_id"],
                            args["group_size"],
                            args["group"],
                            args["Mi_id"],
                            dm_id,
                            args["n_bc"],
                            args["same_alt"],
                            args["D_id"],
                            args["config"],
                            args["Mc_id"],
                            args["P_id"],
                            it,
                            t,
                        )
                        futures_accept[dm_id] = thread_pool.submit(
                            task_thread,
                            tasks_accept,
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
                    dms = [
                        dm_id for dm_id in dms if tasks_P[dm_id].P_file(dir, t).exists()
                    ]

                changes = []
                with task_Mc.C_file(dir).open("r", newline="") as f:
                    C_reader = csv.reader(f, dialect="unix")
                    for row in C_reader:
                        changes.append(int(row[0]))

                it += 1
                task_Mc = CollectiveTask(
                    args["m"],
                    args["n_tr"],
                    args["Atr_id"],
                    args["ko"],
                    args["Mo_id"],
                    args["group_size"],
                    args["group"],
                    args["Mi_id"],
                    args["n_bc"],
                    args["same_alt"],
                    args["D_id"],
                    args["config"],
                    args["Mc_id"],
                    args["P_id"],
                    it,
                )

                with task_Mc.C_file(dir).open("w", newline="") as f:
                    C_writer = csv.writer(f, dialect="unix")

                    for dm_id in range(args["group_size"]):
                        temp = 1
                        while (temp < t) and tasks_P[dm_id].P_file(dir, temp).exists():
                            temp += 1
                        accepted_t = temp - 1

                        copy(
                            tasks_P[dm_id].P_file(dir, accepted_t),
                            task_Mc.D_file(dir, dm_id),
                        )

                        with tasks_P[dm_id].P_file(dir, 0).open("r") as P0f:
                            P0 = from_csv(P0f)
                            with (
                                tasks_P[dm_id].P_file(dir, accepted_t).open("r") as P1f
                            ):
                                P1 = from_csv(P1f)
                                C_writer.writerow([changes[dm_id] + len(P1 - P0)])

                                if dm_id in dms_refusing:
                                    with (
                                        tasks_P[dm_id]
                                        .P_file(dir, accepted_t + 1)
                                        .open("r") as P2f
                                    ):
                                        P2 = from_csv(P2f)

                                        with task_Mc.RP_file(dir, dm_id, it).open(
                                            "w"
                                        ) as f:
                                            to_csv(P2, f)

                                        with task_Mc.RC_file(dir, dm_id, it).open(
                                            "w"
                                        ) as f:
                                            to_csv(P2 - P1, f)

                if not dms_refusing:
                    compromise_found = True
            else:
                futures_clean: list[FutureTaskException] = []
                for dm_id in range(args["group_size"]):
                    task_clean = CleanTask(
                        args["m"],
                        args["n_tr"],
                        args["Atr_id"],
                        args["ko"],
                        args["Mo_id"],
                        args["group_size"],
                        args["group"],
                        args["Mi_id"],
                        dm_id,
                        args["n_bc"],
                        args["same_alt"],
                        args["D_id"],
                        args["config"],
                        args["Mc_id"],
                        args["P_id"],
                        it,
                    )

                    futures_clean.append(
                        thread_pool.submit(
                            task_thread,
                            task_clean,
                            {},
                            task_queue,
                            [future_Mc],
                            dir,
                        )
                    )

                raise_exceptions(futures_clean)

    dir.csv_files["compromise"].queue.put(
        {
            CompromiseFieldnames.M: args["m"],
            CompromiseFieldnames.N_tr: args["n_tr"],
            CompromiseFieldnames.Atr_id: args["Atr_id"],
            CompromiseFieldnames.Ko: args["ko"],
            CompromiseFieldnames.Mo_id: args["Mo_id"],
            CompromiseFieldnames.Group_size: args["group_size"],
            CompromiseFieldnames.Group: args["group"],
            CompromiseFieldnames.Mi_id: args["Mi_id"],
            CompromiseFieldnames.N_bc: args["n_bc"],
            CompromiseFieldnames.Same_alt: args["same_alt"],
            CompromiseFieldnames.D_id: args["D_id"],
            CompromiseFieldnames.Config: args["config"],
            CompromiseFieldnames.Compromise: compromise_found,
            CompromiseFieldnames.Time: max_time - time_left,
            CompromiseFieldnames.It: it,
            CompromiseFieldnames.Changes: sum(changes),
        }
    )
