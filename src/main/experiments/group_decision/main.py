from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product

from ....utils import list_replace
from ...task import FutureTaskException, Task, wait_exception_mapping
from ...threads.task import task_thread
from ...threads.worker_manager import TaskQueue
from ..elicitation.config import MIPConfig
from .arguments import ArgumentsGroupDecision
from .directory import DirectoryGroupDecision
from .seeds import Seeds
from .task import ATask, CollectiveTask, DTask, MieTask, MiTask, MoTask
from .threads.collective import collective_thread


def main(
    args: ArgumentsGroupDecision,
    dir: DirectoryGroupDecision,
    thread_pool: ThreadPoolExecutor,
    task_queue: TaskQueue,
):
    # Constants
    NB_ATR = args.nb_Atr
    NB_MO = args.nb_Mo or NB_ATR
    NB_MI = args.nb_Mi or NB_MO
    NB_D = args.nb_D or NB_MI
    NB_MIE = args.nb_Mie or NB_D
    NB_MC = args.nb_Mc or NB_MIE
    NB_P = args.nb_P or NB_MC

    # Complete seeds
    seeds = Seeds.from_seed(NB_ATR, NB_MO, NB_MI, NB_D, NB_MIE, NB_MC, NB_P, args.seed)
    replace(seeds, A_tr=list_replace(seeds.A_tr, args.seeds.A_tr))
    replace(seeds, Mo=list_replace(seeds.Mo, args.seeds.Mo))
    replace(seeds, Mi=list_replace(seeds.Mi, args.seeds.Mi))
    replace(seeds, D=list_replace(seeds.D, args.seeds.D))
    replace(seeds, Mc=list_replace(seeds.Mc, args.seeds.Mc))
    replace(seeds, P=list_replace(seeds.P, args.seeds.P))

    # Write seeds
    with dir.seeds.open("w") as f:
        f.write(seeds.to_json())

    # Add missing configs
    if not args.config:
        args.config.append(MIPConfig())

    # Write configs
    for config in args.config:
        csv_file = dir.csv_files["configs"]
        csv_file.writerow(
            csv_file.fields(Id=config.id, Method=config.method, Config=config)
        )

    # Write hyperparameters
    for group_parameters in args.group:
        csv_file = dir.csv_files["group_parameters"]
        csv_file.writerow(
            csv_file.fields(
                Id=group_parameters.id,
                Gen=group_parameters.gen,
                Accept=group_parameters.accept,
            )
        )

    # Task dict
    futures: dict[Task, FutureTaskException] = {}

    # Main
    for m in args.M:
        for n_tr, Atr_id in product(args.N_tr, range(NB_ATR)):
            task = ATask(m, n_tr, Atr_id)
            futures[task] = thread_pool.submit(
                task_thread,
                task,
                {"seed": seeds.A_tr[Atr_id]},
                task_queue,
                [],
                dir,
            )

        for ko, Mo_id in product(args.Ko, range(NB_MO)):
            task = MoTask(m, ko, args.fixed_lex_order, Mo_id)
            futures[task] = thread_pool.submit(
                task_thread,
                task,
                {"seed": seeds.Mo[Mo_id]},
                task_queue,
                [],
                dir,
            )

            for group_size, group in product(args.group_size, args.group):
                for Mi_id in range(args.nb_Mi) if args.nb_Mi else [Mo_id]:
                    for dm_id in range(group_size):
                        task = MiTask(
                            m,
                            ko,
                            args.fixed_lex_order,
                            Mo_id,
                            group_size,
                            group,
                            Mi_id,
                            dm_id,
                        )
                        futures[task] = thread_pool.submit(
                            task_thread,
                            task,
                            {"seed": seeds.Mi[Mi_id]},
                            task_queue,
                            [futures[MoTask(m, ko, args.fixed_lex_order, Mo_id)]],
                            dir,
                        )

        for n_tr, ko, group_size, group, n_bc, same_alt in product(
            args.N_tr,
            args.Ko,
            args.group_size,
            args.group,
            args.N_bc,
            args.same_alt,
        ):
            for Atr_id in range(args.nb_Atr):
                for Mo_id in range(args.nb_Mo) if args.nb_Mo else [Atr_id]:
                    for Mi_id in range(args.nb_Mi) if args.nb_Mi else [Mo_id]:
                        for D_id in range(args.nb_D) if args.nb_D else [Mi_id]:
                            for dm_id in range(group_size):
                                task = DTask(
                                    m,
                                    n_tr,
                                    Atr_id,
                                    ko,
                                    args.fixed_lex_order,
                                    Mo_id,
                                    group_size,
                                    group,
                                    Mi_id,
                                    dm_id,
                                    n_bc,
                                    same_alt,
                                    D_id,
                                )
                                futures[task] = thread_pool.submit(
                                    task_thread,
                                    task,
                                    {"seed": seeds.D[D_id]},
                                    task_queue,
                                    [
                                        futures[ATask(m, n_tr, Atr_id)],
                                        futures[
                                            MiTask(
                                                m,
                                                ko,
                                                args.fixed_lex_order,
                                                Mo_id,
                                                group_size,
                                                group,
                                                Mi_id,
                                                dm_id,
                                            )
                                        ],
                                    ],
                                    dir,
                                )

                            for config in args.config:
                                for Mie_id in (
                                    range(args.nb_Mie) if args.nb_Mie else [D_id]
                                ):
                                    for Mie in args.Mie:
                                        precede_futures = [
                                            futures[
                                                DTask(
                                                    m,
                                                    n_tr,
                                                    Atr_id,
                                                    ko,
                                                    args.fixed_lex_order,
                                                    Mo_id,
                                                    group_size,
                                                    group,
                                                    Mi_id,
                                                    dm_id,
                                                    n_bc,
                                                    same_alt,
                                                    D_id,
                                                )
                                            ]
                                            for dm_id in range(group_size)
                                        ]
                                        if Mie:
                                            task = MieTask(
                                                m,
                                                n_tr,
                                                Atr_id,
                                                ko,
                                                args.fixed_lex_order,
                                                Mo_id,
                                                group_size,
                                                group,
                                                Mi_id,
                                                n_bc,
                                                same_alt,
                                                D_id,
                                                config,
                                                Mie_id,
                                            )

                                            futures[task] = thread_pool.submit(
                                                task_thread,
                                                task,
                                                {"seed": seeds.Mie[Mie_id]},
                                                task_queue,
                                                precede_futures,
                                                dir,
                                            )

                                            precede_futures = [
                                                futures[
                                                    MieTask(
                                                        m,
                                                        n_tr,
                                                        Atr_id,
                                                        ko,
                                                        args.fixed_lex_order,
                                                        Mo_id,
                                                        group_size,
                                                        group,
                                                        Mi_id,
                                                        n_bc,
                                                        same_alt,
                                                        D_id,
                                                        config,
                                                        Mie_id,
                                                    )
                                                ]
                                            ]

                                        for path in args.path:
                                            for Mc_id in (
                                                range(args.nb_Mc)
                                                if args.nb_Mc
                                                else [Mie_id]
                                            ):
                                                for P_id in (
                                                    range(args.nb_P)
                                                    if args.nb_P
                                                    else [Mc_id]
                                                ):
                                                    task = CollectiveTask(
                                                        m,
                                                        n_tr,
                                                        Atr_id,
                                                        ko,
                                                        args.fixed_lex_order,
                                                        Mo_id,
                                                        group_size,
                                                        group,
                                                        Mi_id,
                                                        n_bc,
                                                        same_alt,
                                                        D_id,
                                                        config,
                                                        Mie_id,
                                                        Mie,
                                                        Mc_id,
                                                        path,
                                                        P_id,
                                                        0,
                                                    )
                                                    futures[task] = thread_pool.submit(
                                                        collective_thread,
                                                        {
                                                            "max_time": args.max_time,
                                                            "m": m,
                                                            "n_tr": n_tr,
                                                            "ko": ko,
                                                            "Atr_id": Atr_id,
                                                            "Mo_id": Mo_id,
                                                            "Mi_id": Mi_id,
                                                            "Mie_id": Mie_id,
                                                            "Mc_id": Mc_id,
                                                            "P_id": P_id,
                                                            "fixed_lex_order": args.fixed_lex_order,
                                                            "group_size": group_size,
                                                            "group": group,
                                                            "n_bc": n_bc,
                                                            "same_alt": same_alt,
                                                            "D_id": D_id,
                                                            "config": config,
                                                            "path": path,
                                                            "Mie": Mie,
                                                            "seeds": seeds,
                                                        },
                                                        task_queue,
                                                        precede_futures,
                                                        dir,
                                                        args.max_time,
                                                    )

    wait_exception_mapping(futures)
