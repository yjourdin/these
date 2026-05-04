import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from shutil import copy
from typing import Any

from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure

from src.methods import MethodEnum
from src.preference_structure.io import from_csv, to_csv

from ....init_directory import DIR
from ....task import FutureTask, TaskResult, result_dict, result_list
from ....threads.task import task_thread
from ..directory import DirectoryGroupDecision
from ..task import (
    # AcceptMcTask,
    AcceptPTask,
    CleanTask,
    CollectiveMIPTask,
    CollectiveSATask,
    PreferencePathTask,
)


def collective_thread(
    args: dict[str, Any],
    precede_futures: list[FutureTask],
):
    assert isinstance(DIR, DirectoryGroupDecision)
    precede_results = result_list(precede_futures)

    time_passed = 0
    if len(precede_results) == 1:
        time_passed = precede_results[0].time

    with ThreadPoolExecutor() as thread_pool:
        DMS = range(args["group_size"])
        it = 0
        changes: list[int] = [0] * args["group_size"]

        if args["method"] is MethodEnum.MIP:
            task_Mc = CollectiveMIPTask(
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
                args["Mie"],
                args["Mie_config"],
                args["Mie_id"],
                args["config"],
                args["nb_Mcp"],
                args["Mc_id"],
                args["path"],
                args["P_id"],
                it,
            )
        elif args["method"] is MethodEnum.SA:
            task_Mc = CollectiveSATask(
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
                args["Mie_id"],
                args["config"],
                args["nb_Mcp"],
                args["Mc_id"],
                args["path"],
                args["P_id"],
                it,
            )
        else:
            raise TypeError(f"Unknown method : {args['method']}")

        for dm_id in DMS:
            copy(task_Mc.D_file(DIR, dm_id), task_Mc.Di_file(DIR, dm_id))

        with task_Mc.C_file(DIR).open("w", newline="") as f:
            C_writer = csv.writer(f, dialect="unix")
            C_writer.writerows([[0]] * args["group_size"])

        time_left = args["max_time"] - time_passed
        time_left_per_it = args["time_per_it"]
        compromise_found = False
        while (
            (not compromise_found)
            and (time_left >= 1)
            and (
                (not args["Mie"])
                or (
                    isinstance(task_Mc, CollectiveMIPTask)
                    and task_Mc.Mie_file(DIR, 0).exists()
                )
            )
        ):
            future_Mc = thread_pool.submit(
                task_thread,
                task_Mc,
                {
                    "seed": args["seeds"].Mc[args["Mc_id"]],
                    "max_time": min(time_left, time_left_per_it),
                    "nb_cpus": args["nb_cpus"]
                },
                [],
            )

            result_Mc, time_Mc = future_Mc.result()

            time_left -= time_Mc
            time_left_per_it -= time_Mc
            if (time_left < 1) or (time_left_per_it < 1):
                break

            if not result_Mc:
                if args["Mie"] and it == 0:
                    break

                futures_clean: list[FutureTask] = []
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
                        args["Mie"],
                        args["Mie_config"],
                        args["Mie_id"],
                        args["method"],
                        args["config"],
                        args["nb_Mcp"],
                        args["Mc_id"],
                        args["path"],
                        args["P_id"],
                        it,
                    )

                    futures_clean.append(
                        thread_pool.submit(task_thread, task_clean, {}, [])
                    )

                result_list(futures_clean)
            else:
                # futures_accept: dict[int, FutureTask] = {}
                # for dm_id in DMS:
                #     tasks_accept = AcceptMcTask(
                #         args["m"],
                #         args["n_tr"],
                #         args["Atr_id"],
                #         args["ko"],
                #         args["fixed_lex_order"],
                #         args["Mo_id"],
                #         args["group_size"],
                #         args["group"],
                #         args["Mi_id"],
                #         dm_id,
                #         args["n_bc"],
                #         args["same_alt"],
                #         args["D_id"],
                #         args["Mie"],
                #         args["Mie_config"],
                #         args["Mie_id"],
                #         args["method"],
                #         args["config"],
                #         args["nb_Mcp"],
                #         args["Mc_id"],
                #         args["path"],
                #         args["P_id"],
                #         it,
                #     )
                #     futures_accept[dm_id] = thread_pool.submit(
                #         task_thread, tasks_accept, {}, []
                #     )

                # results_accept = result_dict(futures_accept)
                # dms_refusing = [
                #     dm_id for dm_id, result in results_accept.items() if not result.res
                # ]

                # compromise_found = not dms_refusing

                t = 0

                tasks_P: dict[int, PreferencePathTask] = {}
                futures_P: dict[int, FutureTask] = {}
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
                        args["Mie"],
                        args["Mie_config"],
                        args["Mie_id"],
                        args["method"],
                        args["config"],
                        args["nb_Mcp"],
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
                            "max_time": min(time_left, time_left_per_it),
                        },
                        [],
                    )

                results_P = result_list(list(futures_P.values()))

                time_left -= max(result.time for result in results_P)
                time_left_per_it -= max(result.time for result in results_P)
                if (time_left < 1) or (time_left_per_it < 1):
                    break
                if not all(result.res for result in results_P):
                    break

                t = 1
                dms = range(args["group_size"])

                dms_refusing: list[int] = []

                while dms := [
                    dm_id for dm_id in dms if tasks_P[dm_id].P_file(DIR, t).exists()
                ]:
                    futures_accept: dict[int, FutureTask] = {}

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
                            args["Mie"],
                            args["Mie_config"],
                            args["Mie_id"],
                            args["method"],
                            args["config"],
                            args["nb_Mcp"],
                            args["Mc_id"],
                            args["path"],
                            args["P_id"],
                            it,
                            t,
                        )
                        futures_accept[dm_id] = thread_pool.submit(
                            task_thread, tasks_accept, {}, [futures_P[dm_id]]
                        )

                    results_accept = result_dict(futures_accept)

                    dms_refusing = [
                        dm_id
                        for dm_id, result in results_accept.items()
                        if not result.res
                    ]

                    if dms_refusing:
                        break

                    t += 1

                compromise_found = not dms_refusing

                changes = []
                with task_Mc.C_file(DIR).open("r", newline="") as f:
                    C_reader = csv.reader(f, dialect="unix")
                    for row in C_reader:
                        changes.append(int(row[0]))

                new_it = it + 1 if not compromise_found else it
                new_task_Mc = replace(task_Mc, it=new_it)

                for dm_id in DMS:
                    with task_Mc.Di_file(DIR, dm_id=dm_id).open("r") as f:
                        original_D = from_csv(f)

                    temp = 0
                    while (temp < t) and tasks_P[dm_id].P_file(DIR, t=temp).exists():
                        temp += 1
                    accepted_t = temp - 1

                    with (
                        task_Mc.Dc_file(DIR).open("r")
                        if accepted_t == -1
                        else tasks_P[dm_id].P_file(DIR, t=accepted_t).open("r")
                    ) as f:
                        accepted_D = from_csv(f)

                    if not compromise_found:
                        copy(
                            tasks_P[dm_id].P_file(DIR, t=accepted_t),
                            new_task_Mc.Di_file(DIR, dm_id=dm_id),
                        )

                    changes[dm_id] += len(original_D - accepted_D)

                    csv_file = DIR.csv_files["changes"]
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
                            Method=args["method"],
                            Config=args["config"],
                            Mie=args["Mie"],
                            Mie_config=args["Mie_config"],
                            Mie_id=args["Mie_id"],
                            Path=args["path"],
                            P_id=args["P_id"],
                            Mc_id=args["Mc_id"],
                            Nb_Mcp=args["nb_Mcp"],
                            It=it,
                            Dm_id=dm_id,
                            T=accepted_t,
                            Changes=changes[dm_id],
                        )
                    )

                    if not compromise_found:
                        with new_task_Mc.C_file(DIR).open("a", newline="") as f:
                            C_writer = csv.writer(f, dialect="unix")
                            C_writer.writerow([changes[dm_id]])

                        if dm_id in dms_refusing:
                            copy(
                                tasks_P[dm_id].P_file(DIR, accepted_t + 1),
                                new_task_Mc.Dr_file(DIR, dm_id, it),
                            )

                            with (
                                tasks_P[dm_id]
                                .P_file(DIR, accepted_t + 1)
                                .open("r") as f
                            ):
                                refused_D = from_csv(f)

                            Cr: list[Relation] = []
                            for r in refused_D - accepted_D:
                                Cr.append(r)
                                if isinstance(r, I) and (
                                    accepted_r := accepted_D.elements_pairs_relations[
                                        r.a, r.b
                                    ]
                                ):
                                    Cr.append(P(accepted_r.b, accepted_r.a))

                            with (new_task_Mc.Cr_file(DIR, dm_id, it)).open("w") as f:
                                to_csv(PreferenceStructure(Cr, validate=False), f)
                if not compromise_found:
                    it = new_it
                    task_Mc = new_task_Mc
                    time_left_per_it = args["time_per_it"]

    csv_file = DIR.csv_files["compromise"]
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
            Method=args["method"],
            Config=args["config"],
            Mie=args["Mie"],
            Mie_config=args["Mie_config"],
            Mie_id=args["Mie_id"],
            Path=args["path"],
            P_id=args["P_id"],
            Mc_id=args["Mc_id"],
            Nb_Mcp=args["nb_Mcp"],
            Compromise=compromise_found,
            Time=args["max_time"] - time_left,
            It=it + 1,
            Changes=sum(changes),
        )
    )

    return TaskResult(compromise_found, args["max_time"] - time_left)
