import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations
from multiprocessing import Pipe
from multiprocessing.connection import Connection, wait
from shutil import copy
from typing import cast

from src.constants import SENTINEL
from src.methods import MethodEnum
from src.models import ModelEnum
from src.preference_structure.io import from_csv, to_csv

from .....utils import catchtime
from ....dir import DIR
from ....task import FutureTask, TaskResult, result_dict, result_list
from ....threads.task import task_thread
from ...elicitation.config import Config, MIPConfig, SAConfig
from ..directory import DirectoryGroupDecision
from ..fields import GroupParameters
from ..seeds import Seeds
from ..task import (
    # AcceptMcTask,
    AcceptPTask,
    # CleanTask,
    CollectiveMIPTask,
    CollectiveSATask,
    DistanceTask,
    PreferencePathTask,
)


def collective_thread(
    m: int,
    n_tr: int,
    Atr_id: int,
    model: ModelEnum,
    ko: int,
    fixed_lex_order: bool,
    Mo_id: int,
    group_size: int,
    group: GroupParameters,
    Mi_id: int,
    n_bc: int,
    same_alt: bool,
    D_id: int,
    Mie: bool,
    Mie_config: MIPConfig | None,
    Mie_id: int,
    method: MethodEnum,
    config: Config,
    nb_Mcp: int,
    Mc_id: int,
    path: bool,
    P_id: int,
    seeds: Seeds,
    max_time: int,
    time_per_it: int,
    precede_futures: list[FutureTask],
):
    assert isinstance(DIR, DirectoryGroupDecision)
    precede_results = result_list(precede_futures)

    time_passed = 0
    if len(precede_results) == 1:
        time_passed = precede_results[0].time

    with ThreadPoolExecutor() as thread_pool:
        DMS = range(group_size)
        it = 0
        changes: list[int] = [0] * group_size

        if method is MethodEnum.MIP:
            assert isinstance(config, MIPConfig)
            task_Mc = CollectiveMIPTask(
                m,
                n_tr,
                Atr_id,
                model,
                ko,
                fixed_lex_order,
                Mo_id,
                group_size,
                group,
                Mi_id,
                n_bc,
                same_alt,
                D_id,
                Mie,
                Mie_config,
                Mie_id,
                config,
                nb_Mcp,
                Mc_id,
                path,
                P_id,
                it,
            )
        elif method is MethodEnum.SA:
            assert isinstance(config, SAConfig)
            task_Mc = CollectiveSATask(
                m,
                n_tr,
                Atr_id,
                model,
                ko,
                fixed_lex_order,
                Mo_id,
                group_size,
                group,
                Mi_id,
                n_bc,
                same_alt,
                D_id,
                Mie_id,
                config,
                nb_Mcp,
                Mc_id,
                path,
                P_id,
                it,
            )

        for dm_id in DMS:
            copy(task_Mc.D_file(DIR, dm_id), task_Mc.Di_file(DIR, dm_id))

        with task_Mc.C_file(DIR).open("w", newline="") as f:
            C_writer = csv.writer(f, "unix")
            C_writer.writerows([[0]] * group_size)

        time_left = max_time - time_passed
        time_left_per_it = time_per_it
        compromise_found = False
        while (
            (not compromise_found)
            and (time_left >= 1)
            and (
                (not Mie)
                or (
                    isinstance(task_Mc, CollectiveMIPTask)
                    and task_Mc.Mie_file(DIR, 0).exists()
                )
            )
        ):
            future_Mc = thread_pool.submit(
                task_thread,
                task_Mc,  # pyright: ignore[reportUnknownArgumentType]
                seed=seeds.Mc[Mc_id],
                max_time=min(time_left, time_left_per_it),
                nb_cpus=config.nb_cpus,
            )

            result_Mc, time_Mc = future_Mc.result()
            time_left -= time_Mc
            time_left_per_it -= time_Mc
            if time_left < 1:  # or (time_left_per_it < 1):
                # raise ValueError("Time left")
                break
            # return TaskResult(result_Mc, max_time - time_left)

            if not result_Mc:
                # raise ValueError("Mc")
                break
                # if Mie and it == 0:
                #     break

                # futures_clean: list[FutureTask] = []
                # for dm_id in DMS:
                #     task_clean = CleanTask(
                #         m,
                #         n_tr,
                #         Atr_id,
                #         ko,
                #         fixed_lex_order,
                #         Mo_id,
                #         group_size,
                #         group,
                #         Mi_id,
                #         dm_id,
                #         n_bc,
                #         same_alt,
                #         D_id,
                #         Mie,
                #         Mie_config,
                #         Mie_id,
                #         method,
                #         config,
                #         nb_Mcp,
                #         Mc_id,
                #         path,
                #         P_id,
                #         it,
                #     )

                #     futures_clean.append(
                #         thread_pool.submit(task_thread, task_clean, {}, [])
                #     )

                # result_list(futures_clean)
            else:
                for a, b in combinations(range(nb_Mcp), 2):
                    task = DistanceTask(
                        m,
                        n_tr,
                        Atr_id,
                        model,
                        ko,
                        fixed_lex_order,
                        Mo_id,
                        group_size,
                        group,
                        Mi_id,
                        n_bc,
                        same_alt,
                        D_id,
                        Mie,
                        Mie_config,
                        Mie_id,
                        method,
                        config,
                        nb_Mcp,
                        Mc_id,
                        path,
                        P_id,
                        it,
                        a,
                        b,
                    )
                    thread_pool.submit(task_thread, task)

                # futures_accept: dict[int, FutureTask] = {}
                # for dm_id in DMS:
                #     tasks_accept = AcceptMcTask(
                #         m,
                #         n_tr,
                #         Atr_id,
                #         ko,
                #         fixed_lex_order,
                #         Mo_id,
                #         group_size,
                #         group,
                #         Mi_id,
                #         dm_id,
                #         n_bc,
                #         same_alt,
                #         D_id,
                #         Mie,
                #         Mie_config,
                #         Mie_id,
                #         method,
                #         config,
                #         nb_Mcp,
                #         Mc_id,
                #         path,
                #         P_id,
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

                # tasks_P: dict[int, PreferencePathTask] = {}
                # futures_P: dict[int, FutureTask] = {}
                # for dm_id in DMS:
                #     tasks_P[dm_id] = PreferencePathTask(
                #         m,
                #         n_tr,
                #         Atr_id,
                #         model,
                #         ko,
                #         fixed_lex_order,
                #         Mo_id,
                #         group_size,
                #         group,
                #         Mi_id,
                #         dm_id,
                #         n_bc,
                #         same_alt,
                #         D_id,
                #         Mie,
                #         Mie_config,
                #         Mie_id,
                #         method,
                #         config,
                #         nb_Mcp,
                #         Mc_id,
                #         path,
                #         P_id,
                #         it,
                #     )

                #     futures_P[dm_id] = thread_pool.submit(
                #         task_thread,
                #         tasks_P[dm_id],
                #         {
                #             "seed": seeds.P[P_id],
                #             "max_time": min(time_left, time_left_per_it),
                #         },
                #         [],
                #     )

                # results_P = result_list(list(futures_P.values()))

                # time_left -= max(result.time for result in results_P)
                # time_left_per_it -= max(result.time for result in results_P)
                # if time_left < 1:  # or (time_left_per_it < 1):
                #     break
                # if not all(result.res for result in results_P):
                #     break

                futures_P: dict[int, FutureTask] = {}
                sources: dict[Connection, set[int]] = {}

                for dm_id in DMS:
                    main_connection, worker_connection = Pipe()
                    sources[main_connection] = set()
                    task = PreferencePathTask(
                        m,
                        n_tr,
                        Atr_id,
                        model,
                        ko,
                        fixed_lex_order,
                        Mo_id,
                        group_size,
                        group,
                        Mi_id,
                        dm_id,
                        n_bc,
                        same_alt,
                        D_id,
                        Mie,
                        Mie_config,
                        Mie_id,
                        method,
                        config,
                        nb_Mcp,
                        Mc_id,
                        path,
                        P_id,
                        it,
                    )
                    futures_P[dm_id] = thread_pool.submit(
                        task_thread,
                        task,
                        seed=seeds.P[P_id],
                        max_time=min(time_left, time_left_per_it),
                        connection=worker_connection,
                    )

                working_connections = list(sources.keys())

                with catchtime() as time:
                    while working_connections and not set.intersection(
                        *sources.values()
                    ):
                        for connection in cast(
                            list[Connection], wait(working_connections)
                        ):
                            if (source := connection.recv()) != SENTINEL:
                                sources[connection] = sources[connection] | source
                            else:
                                working_connections.remove(connection)
                                if not set.intersection(*[
                                    sources[connection]
                                    for connection in sources
                                    if connection not in working_connections
                                ]):
                                    working_connections = []

                    source = (
                        intersect.pop()
                        if (
                            working_connections
                            and (intersect := set.intersection(*sources.values()))
                        )
                        else SENTINEL
                    )
                    for connection in sources:
                        connection.send(source)

                    results_P = result_list(list(futures_P.values()))

                time_left -= time()
                time_left_per_it -= time()
                if time_left < 1:  # or (time_left_per_it < 1):
                    # raise ValueError("Time left 2")
                    break
                if not all(result.res for result in results_P):
                    # raise ValueError("Preference path")
                    break

                tasks_accept: dict[int, AcceptPTask] = {}
                futures_accept: dict[int, FutureTask] = {}
                for dm_id in DMS:
                    tasks_accept[dm_id] = AcceptPTask(
                        m,
                        n_tr,
                        Atr_id,
                        model,
                        ko,
                        fixed_lex_order,
                        Mo_id,
                        group_size,
                        group,
                        Mi_id,
                        dm_id,
                        n_bc,
                        same_alt,
                        D_id,
                        Mie,
                        Mie_config,
                        Mie_id,
                        method,
                        config,
                        nb_Mcp,
                        Mc_id,
                        path,
                        P_id,
                        it,
                    )
                    futures_accept[dm_id] = thread_pool.submit(
                        task_thread,
                        tasks_accept[dm_id],
                        precede_futures=[futures_P[dm_id]],
                    )

                results_accept = result_dict(futures_accept)

                if all((result.res == -1) for result in results_accept.values()):
                    compromise_found = True
                    t = None
                else:
                    compromise_found = False

                    t = min(
                        int(result.res)
                        for result in results_accept.values()
                        if result.res >= 0
                    )

                    # dms_refusing = [
                    #     dm_id
                    #     for dm_id, result in results_accept.items()
                    #     if result.res == t
                    # ]

                changes = []
                with task_Mc.C_file(DIR).open("r", newline="") as f:
                    C_reader = csv.reader(f, dialect="unix")  # pyright: ignore[reportUnknownArgumentType]
                    for row in C_reader:
                        changes.append(int(row[0]))

                new_task_Mc = (
                    replace(task_Mc, it=it + 1) if not compromise_found else None  # pyright: ignore[reportUnknownArgumentType]
                )

                if new_task_Mc is not None:
                    copy(task_Mc.Cr_file(DIR), new_task_Mc.Cr_file(DIR))  # pyright: ignore[reportUnknownArgumentType]

                for dm_id in DMS:
                    with tasks_accept[dm_id].Di_file(DIR).open("r") as f:
                        D = from_csv(f)

                    with tasks_accept[dm_id].P_file(DIR).open("r") as f:
                        P = from_csv(f)

                    t_dm = min(t, len(P)) if t is not None else len(P)

                    changes[dm_id] += t_dm

                    csv_file = DIR.csv_files["changes"]
                    csv_file.writerow(
                        M=m,
                        N_tr=n_tr,
                        Atr_id=Atr_id,
                        Ko=ko,
                        Mo_id=Mo_id,
                        Group_size=group_size,
                        Group=group,
                        Mi_id=Mi_id,
                        N_bc=n_bc,
                        Same_alt=same_alt,
                        D_id=D_id,
                        Method=method,
                        Config=config,
                        Mie=Mie,
                        Mie_config=Mie_config,
                        Mie_id=Mie_id,
                        Path=path,
                        P_id=P_id,
                        Mc_id=Mc_id,
                        Nb_Mcp=nb_Mcp,
                        It=it,
                        Dm_id=dm_id,
                        T=t_dm,
                        Changes=changes[dm_id],
                    )

                    if new_task_Mc is not None:
                        for i in range(t_dm):
                            new_relation = P.relations[i]
                            if old_relation := D.elements_pairs_relations.get(
                                new_relation.elements
                            ):
                                D -= old_relation
                            D += new_relation

                        with new_task_Mc.Di_file(DIR, dm_id=dm_id).open("w") as f:
                            to_csv(D, f)

                        with new_task_Mc.C_file(DIR).open("a", newline="") as f:
                            C_writer = csv.writer(f, "unix")  # pyright: ignore[reportUnknownArgumentType]
                            C_writer.writerow([changes[dm_id]])

                        # if dm_id in dms_refusing:
                        #     with new_task_Mc.Dr_file(DIR).open("a") as f:
                        #         to_csv(PreferenceStructure(P.relations[t_dm]), f)

                if new_task_Mc is not None:
                    it = it + 1
                    task_Mc = new_task_Mc
                    time_left_per_it = time_per_it

    csv_file = DIR.csv_files["compromise"]
    csv_file.writerow(
        M=m,
        N_tr=n_tr,
        Atr_id=Atr_id,
        Model=model,
        Ko=ko,
        Mo_id=Mo_id,
        Group_size=group_size,
        Group=group,
        Mi_id=Mi_id,
        N_bc=n_bc,
        Same_alt=same_alt,
        D_id=D_id,
        Method=method,
        Config=config,
        Mie=Mie,
        Mie_config=Mie_config,
        Mie_id=Mie_id,
        Path=path,
        P_id=P_id,
        Mc_id=Mc_id,
        Nb_Mcp=nb_Mcp,
        Compromise=compromise_found,
        Time=max_time - time_left,
        It=it + 1,
        Changes=sum(changes),
    )

    return TaskResult(compromise_found, max_time - time_left)
