from concurrent.futures import ThreadPoolExecutor
from itertools import product

from ....utils import list_replace
from ...task import FutureTaskException, Task
from ...threads.task import task_thread
from ...threads.worker_manager import TaskQueue
from ..elicitation.config import MIPConfig
from ..elicitation.fieldnames import ConfigFieldnames
from .arguments import ArgumentsGroupDecision
from .directory import DirectoryGroupDecision
from .fieldnames import GroupParametersFieldnames
from .seeds import Seeds
from .task import (
    ATrainTask,
    DTask,
    MiTask,
    MoTask,
)
from .threads.collective import collective_thread
from .threads.collective_no_path import collective_no_path_thread


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
    NB_MC = args.nb_Mc or NB_D
    NB_P = args.nb_P or NB_MC

    # Complete seeds
    seeds = Seeds.from_seed(NB_ATR, NB_MO, NB_MI, NB_D, NB_MC, NB_P, args.seed)
    list_replace(seeds.A_tr, args.seeds.A_tr)
    list_replace(seeds.Mo, args.seeds.Mo)
    list_replace(seeds.Mi, args.seeds.Mi)
    list_replace(seeds.D, args.seeds.D)
    list_replace(seeds.Mc, args.seeds.Mc)
    list_replace(seeds.P, args.seeds.P)

    # Add missing configs
    if not args.config:
        args.config.append(MIPConfig())

    # Write configs
    for config in args.config:
        dir.csv_files["configs"].queue.put(
            {
                ConfigFieldnames.Id: config.id,
                ConfigFieldnames.Method: config.method,
                ConfigFieldnames.Config: {
                    k: v for k, v in config.to_dict().items() if k != "id"
                },
            }
        )

    # Write hyperparameters
    for group_parameters in args.group:
        dir.csv_files["group_parameters"].queue.put(
            {
                GroupParametersFieldnames.Id: group_parameters.id,
                GroupParametersFieldnames.Gen: group_parameters.gen,
                GroupParametersFieldnames.Accept: group_parameters.accept,
            }
        )

    # Task dict
    futures: dict[Task, FutureTaskException] = {}

    # Main
    for m in args.M:
        for n_tr, Atr_id in product(args.N_tr, range(NB_ATR)):
            task = ATrainTask(m, n_tr, Atr_id)
            futures[task] = thread_pool.submit(
                task_thread,
                task,
                {"seed": seeds.A_tr[Atr_id]},
                task_queue,
                [],
                dir,
            )

        for ko, Mo_id in product(args.Ko, range(NB_MO)):
            task = MoTask(m, ko, Mo_id, args.fixed_lex_order)
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
                            Mo_id,
                            args.fixed_lex_order,
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
                            [futures[MoTask(m, ko, Mo_id, args.fixed_lex_order)]],
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
                                    Mo_id,
                                    args.fixed_lex_order,
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
                                        futures[ATrainTask(m, n_tr, Atr_id)],
                                        futures[
                                            MiTask(
                                                m,
                                                ko,
                                                Mo_id,
                                                args.fixed_lex_order,
                                                group_size,
                                                group,
                                                Mi_id,
                                                dm_id,
                                            )
                                        ],
                                    ],
                                    dir,
                                )

                            for config, path in product(args.config, args.path):
                                for Mc_id in (
                                    range(args.nb_Mc) if args.nb_Mc else [D_id]
                                ):
                                    if path:
                                        for P_id in (
                                            range(args.nb_P) if args.nb_P else [Mc_id]
                                        ):
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
                                                    "Mc_id": Mc_id,
                                                    "P_id": P_id,
                                                    "fixed_lex_order": args.fixed_lex_order,
                                                    "group_size": group_size,
                                                    "group": group,
                                                    "n_bc": n_bc,
                                                    "same_alt": same_alt,
                                                    "D_id": D_id,
                                                    "config": config,
                                                    "seeds": seeds,
                                                },
                                                task_queue,
                                                [
                                                    futures[
                                                        DTask(
                                                            m,
                                                            n_tr,
                                                            Atr_id,
                                                            ko,
                                                            Mo_id,
                                                            args.fixed_lex_order,
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
                                                ],
                                                dir,
                                                args.max_time,
                                            )
                                    else:
                                        futures[task] = thread_pool.submit(
                                            collective_no_path_thread,
                                            {
                                                "max_time": args.max_time,
                                                "m": m,
                                                "n_tr": n_tr,
                                                "ko": ko,
                                                "Atr_id": Atr_id,
                                                "Mo_id": Mo_id,
                                                "Mi_id": Mi_id,
                                                "Mc_id": Mc_id,
                                                "P_id": P_id,
                                                "group_size": group_size,
                                                "group": group,
                                                "n_bc": n_bc,
                                                "same_alt": same_alt,
                                                "D_id": D_id,
                                                "config": config,
                                                "seeds": seeds,
                                            },
                                            task_queue,
                                            [
                                                futures[
                                                    DTask(
                                                        m,
                                                        n_tr,
                                                        Atr_id,
                                                        ko,
                                                        Mo_id,
                                                        args.fixed_lex_order,
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
                                            ],
                                            dir,
                                            args.max_time,
                                        )
