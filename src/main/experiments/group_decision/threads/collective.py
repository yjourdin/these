import csv
from concurrent.futures import ThreadPoolExecutor
from shutil import copy
from typing import Any

from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure

from .....constants import DEFAULT_MAX_TIME
from .....preference_structure.io import from_csv, to_csv
from ....task import (
    FutureTaskException,
    wait_exception,
    wait_exception_iterable,
    wait_exception_mapping,
)
from ....threads.task import task_thread
from ....threads.worker_manager import TaskQueue
from ..directory import DirectoryGroupDecision
from ..task import (
    AcceptMcTask,
    AcceptPTask,
    CleanTask,
    CollectiveTask,
    PreferencePathTask,
)


def collective_thread(
    args: dict[str, Any],
    task_queue: TaskQueue,
    precede_futures: list[FutureTaskException],
    dir: DirectoryGroupDecision,
    max_time: int = DEFAULT_MAX_TIME,
):
    # print(precede_futures, args)
    wait_exception_iterable(precede_futures)
    time_passed = 0
    if len(precede_futures) == 1:
        future = precede_futures[0]
        time_passed = future.result().time if wait_exception(future) else 0

    with ThreadPoolExecutor() as thread_pool:
        DMS = range(args["group_size"])
        it = 0
        changes: list[int] = [0] * args["group_size"]

        task_Mc = CollectiveTask(
            args["m"],
            args["n_tr"],
            args["Atr_id"],
            args["ko"],
            args["fixed_lex_order"],
            args["Mo_id"],
            args["group_size"],
            args["group"],
            args["Mi_id"],
            args["n_bc"],
            args["same_alt"],
            args["D_id"],
            args["config"],
            args["Mie_id"],
            args["Mie"],
            args["Mc_id"],
            args["path"],
            args["P_id"],
            it,
        )

        for dm_id in DMS:
            copy(task_Mc.D_file(dir, dm_id), task_Mc.Di_file(dir, dm_id))

        with task_Mc.C_file(dir).open("w", newline="") as f:
            C_writer = csv.writer(f, dialect="unix")
            C_writer.writerows([[0]] * args["group_size"])

        time_left = max_time - time_passed
        compromise_found = False
        while (
            (not compromise_found)
            and (time_left >= 1)
            and ((not args["Mie"]) or (task_Mc.Mie_file(dir, 0).exists()))
        ):
            future_Mc = thread_pool.submit(
                task_thread,
                task_Mc,
                {"seed": args["seeds"].Mc[args["Mc_id"]], "max_time": time_left},
                task_queue,
                [],
                dir,
            )

            if wait_exception(future_Mc):
                result, time = future_Mc.result()

                time_left -= time
                if time_left < 1:
                    break

                if not result:
                    if args["Mie"] and it == 0:
                        break

                    futures_clean: list[FutureTaskException] = []
                    for dm_id in DMS:
                        task_clean = CleanTask(
                            args["m"],
                            args["n_tr"],
                            args["Atr_id"],
                            args["ko"],
                            args["fixed_lex_order"],
                            args["Mo_id"],
                            args["group_size"],
                            args["group"],
                            args["Mi_id"],
                            dm_id,
                            args["n_bc"],
                            args["same_alt"],
                            args["D_id"],
                            args["config"],
                            args["Mie_id"],
                            args["Mie"],
                            args["Mc_id"],
                            args["path"],
                            args["P_id"],
                            it,
                        )

                        futures_clean.append(
                            thread_pool.submit(
                                task_thread,
                                task_clean,
                                {},
                                task_queue,
                                [],
                                dir,
                            )
                        )

                    wait_exception_iterable(futures_clean)
                else:
                    futures_accept: dict[int, FutureTaskException] = {}
                    for dm_id in DMS:
                        tasks_accept = AcceptMcTask(
                            args["m"],
                            args["n_tr"],
                            args["Atr_id"],
                            args["ko"],
                            args["fixed_lex_order"],
                            args["Mo_id"],
                            args["group_size"],
                            args["group"],
                            args["Mi_id"],
                            dm_id,
                            args["n_bc"],
                            args["same_alt"],
                            args["D_id"],
                            args["config"],
                            args["Mie_id"],
                            args["Mie"],
                            args["Mc_id"],
                            args["path"],
                            args["P_id"],
                            it,
                        )
                        futures_accept[dm_id] = thread_pool.submit(
                            task_thread,
                            tasks_accept,
                            {},
                            task_queue,
                            [],
                            dir,
                        )

                    dms_refusing = []
                    if wait_exception_mapping(futures_accept):
                        dms_refusing = [
                            dm_id
                            for dm_id, future in futures_accept.items()
                            if not future.result().result
                        ]

                    compromise_found = not dms_refusing

                    t = 0

                    if not compromise_found:
                        tasks_P: dict[int, PreferencePathTask] = {}
                        futures_P: dict[int, FutureTaskException] = {}
                        for dm_id in DMS:
                            tasks_P[dm_id] = PreferencePathTask(
                                args["m"],
                                args["n_tr"],
                                args["Atr_id"],
                                args["ko"],
                                args["fixed_lex_order"],
                                args["Mo_id"],
                                args["group_size"],
                                args["group"],
                                args["Mi_id"],
                                dm_id,
                                args["n_bc"],
                                args["same_alt"],
                                args["D_id"],
                                args["config"],
                                args["Mie_id"],
                                args["Mie"],
                                args["Mc_id"],
                                args["path"],
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
                                [],
                                dir,
                            )

                        futures_P_values = futures_P.values()
                        if wait_exception_iterable(futures_P_values):
                            # for k, future in futures_P.items():
                            #     if future.result() is None:
                            #         print(future.done() , args, k)
                            time_left -= max(
                                future.result().time for future in futures_P_values
                            )
                            if time_left < 1:
                                break
                            if not all(
                                future.result().result for future in futures_P_values
                            ):
                                break

                        t = 1
                        dms = range(args["group_size"])

                        dms_refusing: list[int] = []

                        while dms := [
                            dm_id
                            for dm_id in dms
                            if tasks_P[dm_id].P_file(dir, t).exists()
                        ]:
                            futures_accept: dict[int, FutureTaskException] = {}

                            for dm_id in dms:
                                tasks_accept = AcceptPTask(
                                    args["m"],
                                    args["n_tr"],
                                    args["Atr_id"],
                                    args["ko"],
                                    args["fixed_lex_order"],
                                    args["Mo_id"],
                                    args["group_size"],
                                    args["group"],
                                    args["Mi_id"],
                                    dm_id,
                                    args["n_bc"],
                                    args["same_alt"],
                                    args["D_id"],
                                    args["config"],
                                    args["Mie_id"],
                                    args["Mie"],
                                    args["Mc_id"],
                                    args["path"],
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

                            if wait_exception_mapping(futures_accept):
                                dms_refusing = [
                                    dm_id
                                    for dm_id, future in futures_accept.items()
                                    if not future.result().result
                                ]

                            if dms_refusing:
                                break

                            t += 1

                        if not dms_refusing:
                            compromise_found = True

                    changes = []
                    with task_Mc.C_file(dir).open("r", newline="") as f:
                        C_reader = csv.reader(f, dialect="unix")
                        for row in C_reader:
                            changes.append(int(row[0]))

                    if not compromise_found:
                        new_it = it + 1
                        new_task_Mc = CollectiveTask(
                            args["m"],
                            args["n_tr"],
                            args["Atr_id"],
                            args["ko"],
                            args["fixed_lex_order"],
                            args["Mo_id"],
                            args["group_size"],
                            args["group"],
                            args["Mi_id"],
                            args["n_bc"],
                            args["same_alt"],
                            args["D_id"],
                            args["config"],
                            args["Mie_id"],
                            args["Mie"],
                            args["Mc_id"],
                            args["path"],
                            args["P_id"],
                            new_it,
                        )

                    for dm_id in DMS:
                        with task_Mc.Di_file(dir, dm_id).open("r") as f:
                            original_D = from_csv(f)

                        temp = 0
                        while (temp < t) and tasks_P[dm_id].P_file(dir, temp).exists():
                            temp += 1
                        accepted_t = temp - 1

                        with (
                            task_Mc.Dc_file(dir).open("r")
                            if accepted_t == -1
                            else tasks_P[dm_id].P_file(dir, accepted_t).open("r")
                        ) as f:
                            accepted_D = from_csv(f)

                        if not compromise_found:
                            copy(
                                tasks_P[dm_id].P_file(dir, accepted_t),
                                new_task_Mc.Di_file(dir, dm_id),
                            )

                        changes[dm_id] += len(original_D - accepted_D)

                        csv_file = dir.csv_files["changes"]
                        csv_file.writerow(
                            csv_file.fields(
                                M=args["m"],
                                N_tr=args["n_tr"],
                                Atr_id=args["Atr_id"],
                                Ko=args["ko"],
                                Mo_id=args["Mo_id"],
                                Group_size=args["group_size"],
                                Group=args["group"],
                                Mi_id=args["Mi_id"],
                                N_bc=args["n_bc"],
                                Same_alt=args["same_alt"],
                                D_id=args["D_id"],
                                Config=args["config"],
                                Mie=args["Mie"],
                                Mie_id=args["Mie_id"],
                                Path=args["path"],
                                P_id=args["P_id"],
                                Mc_id=args["Mc_id"],
                                It=it,
                                Dm_id=dm_id,
                                T=accepted_t,
                                Changes=changes[dm_id],
                            )
                        )

                        if not compromise_found:
                            with new_task_Mc.C_file(dir).open("a", newline="") as f:
                                C_writer = csv.writer(f, dialect="unix")
                                C_writer.writerow([changes[dm_id]])

                            if dm_id in dms_refusing:
                                copy(
                                    tasks_P[dm_id].P_file(dir, accepted_t + 1),
                                    new_task_Mc.Dr_file(dir, dm_id, it),
                                )

                                with (
                                    tasks_P[dm_id]
                                    .P_file(dir, accepted_t + 1)
                                    .open("r") as f
                                ):
                                    refused_D = from_csv(f)

                                Cr: list[Relation] = []
                                for r in refused_D - accepted_D:
                                    Cr.append(r)
                                    if isinstance(r, I):
                                        if (
                                            accepted_r
                                            := accepted_D.elements_pairs_relations[
                                                r.a, r.b
                                            ]
                                        ):
                                            Cr.append(P(accepted_r.b, accepted_r.a))

                                with (new_task_Mc.Cr_file(dir, dm_id, it)).open(
                                    "w"
                                ) as f:
                                    to_csv(PreferenceStructure(Cr, validate=False), f)
                    if not compromise_found:
                        it = new_it
                        task_Mc = new_task_Mc

    csv_file = dir.csv_files["compromise"]
    csv_file.writerow(
        csv_file.fields(
            M=args["m"],
            N_tr=args["n_tr"],
            Atr_id=args["Atr_id"],
            Ko=args["ko"],
            Mo_id=args["Mo_id"],
            Group_size=args["group_size"],
            Group=args["group"],
            Mi_id=args["Mi_id"],
            N_bc=args["n_bc"],
            Same_alt=args["same_alt"],
            D_id=args["D_id"],
            Config=args["config"],
            Mie=args["Mie"],
            Mie_id=args["Mie_id"],
            Path=args["path"],
            P_id=args["P_id"],
            Mc_id=args["Mc_id"],
            Compromise=compromise_found,
            Time=max_time - time_left,
            It=it + 1,
            Changes=sum(changes),
        )
    )
