from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import cast

from ....methods import MethodEnum
from ....models import ModelEnum
from ....utils import list_replace
from ...task import FutureTaskException, Task, wait_exception_mapping
from ...threads.task import task_thread
from ...threads.worker_manager import TaskQueue
from .arguments import ArgumentsElicitation
from .config import MIPConfig, SAConfig, create_config
from .directory import DirectoryElicitation
from .seeds import Seeds
from .task import ATestTask, ATrainTask, DTask, MIPTask, MoTask, SATask, TestTask


def main(
    args: ArgumentsElicitation,
    dir: DirectoryElicitation,
    thread_pool: ThreadPoolExecutor,
    task_queue: TaskQueue,
):
    # Constants
    NB_ATR = args.nb_Atr
    NB_MO = args.nb_Mo or NB_ATR
    NB_D = args.nb_D or NB_MO
    NB_ME = args.nb_Me or NB_D
    NB_ATE = args.nb_Ate or NB_ME

    # Complete seeds
    seeds = Seeds.from_seed(NB_ATR, NB_ATE, NB_MO, NB_D, NB_ME, args.seed)
    list_replace(seeds.A_tr, args.seeds.A_tr)
    list_replace(seeds.A_te, args.seeds.A_te)
    list_replace(seeds.Mo, args.seeds.Mo)
    list_replace(seeds.D, args.seeds.D)
    list_replace(seeds.Me, args.seeds.Me)

    # Write seeds
    with dir.seeds.open("w") as f:
        f.write(seeds.to_json())

    # Add missing configs
    for method in args.method:
        if not any(config.method is method for config in args.config):
            args.config.append(create_config(method=method))

    # Write configs
    for config in args.config:
        csv_file = dir.csv_files["configs"]
        csv_file.writerow(
            csv_file.fields(Id=config.id, Method=config.method, Config=config)
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

        for n_te, Ate_id in product(
            args.N_te if args.N_te else args.N_tr, range(NB_ATE)
        ):
            task = ATestTask(m, n_te, Ate_id)
            futures[task] = thread_pool.submit(
                task_thread,
                task,
                {"seed": seeds.A_te[Ate_id]},
                task_queue,
                [],
                dir,
            )

        for Mo, ko, group_size, Mo_id in product(
            args.Mo, args.Ko, args.group_size, range(NB_MO)
        ):
            task = MoTask(m, Mo, ko, group_size, args.fixed_lex_order, Mo_id)
            futures[task] = thread_pool.submit(
                task_thread,
                task,
                {"seed": seeds.Mo[Mo_id]},
                task_queue,
                [],
                dir,
            )

        for n_tr, Mo, group_size, ko, n_bc, same_alt, error in product(
            args.N_tr,
            args.Mo,
            args.group_size,
            args.Ko if Mo.value[0] in (ModelEnum.RMP, ModelEnum.SRMP) else [0],
            args.N_bc,
            args.same_alt,
            args.error,
        ):
            for Atr_id in range(args.nb_Atr):
                for Mo_id in range(args.nb_Mo) if args.nb_Mo else [Atr_id]:
                    for D_id in range(args.nb_D) if args.nb_D else [Mo_id]:
                        for dm_id in range(group_size):
                            task = DTask(
                                m,
                                n_tr,
                                Atr_id,
                                Mo,
                                ko,
                                group_size,
                                args.fixed_lex_order,
                                Mo_id,
                                n_bc,
                                same_alt,
                                error,
                                D_id,
                                dm_id,
                            )
                            futures[task] = thread_pool.submit(
                                task_thread,
                                task,
                                {"seed": seeds.D[D_id]},
                                task_queue,
                                [
                                    futures[ATrainTask(m, n_tr, Atr_id)],
                                    futures[
                                        MoTask(
                                            m,
                                            Mo,
                                            ko,
                                            group_size,
                                            args.fixed_lex_order,
                                            Mo_id,
                                        )
                                    ],
                                ],
                                dir,
                            )

                        for Me, ke, method, Me_id in product(
                            args.Me if args.Me else [Mo],
                            args.Ke if (not args.fixed_lex_order and args.Ke) else [ko],
                            args.method,
                            range(args.nb_Me) if args.nb_Me else [D_id],
                        ):
                            for config in (
                                config
                                for config in args.config
                                if config.method is method
                            ):
                                task_Me: Task
                                match method:
                                    case MethodEnum.SA:
                                        task_Me = SATask(
                                            m,
                                            n_tr,
                                            Atr_id,
                                            Mo,
                                            ko,
                                            group_size,
                                            args.fixed_lex_order,
                                            Mo_id,
                                            n_bc,
                                            same_alt,
                                            error,
                                            D_id,
                                            Me,
                                            ke,
                                            cast(SAConfig, config),
                                            Me_id,
                                        )
                                    case MethodEnum.MIP if (
                                        Me.value[0] is ModelEnum.SRMP
                                    ):
                                        task_Me = MIPTask(
                                            m,
                                            n_tr,
                                            Atr_id,
                                            Mo,
                                            ko,
                                            group_size,
                                            args.fixed_lex_order,
                                            Mo_id,
                                            n_bc,
                                            same_alt,
                                            error,
                                            D_id,
                                            Me,
                                            ke,
                                            cast(MIPConfig, config),
                                            Me_id,
                                        )
                                    case _:
                                        break
                                futures[task_Me] = thread_pool.submit(
                                    task_thread,
                                    task_Me,
                                    {"seed": seeds.Me[Me_id]},
                                    task_queue,
                                    [
                                        futures[
                                            DTask(
                                                m,
                                                n_tr,
                                                Atr_id,
                                                Mo,
                                                ko,
                                                group_size,
                                                args.fixed_lex_order,
                                                Mo_id,
                                                n_bc,
                                                same_alt,
                                                error,
                                                D_id,
                                                dm_id,
                                            )
                                        ]
                                        for dm_id in range(group_size)
                                    ],
                                    dir,
                                )

                                for n_te, Ate_id in product(
                                    args.N_te if args.N_te else [n_tr],
                                    range(args.nb_Ate) if args.nb_Ate else [Me_id],
                                ):
                                    task = TestTask(
                                        m,
                                        n_tr,
                                        Atr_id,
                                        Mo,
                                        ko,
                                        group_size,
                                        args.fixed_lex_order,
                                        Mo_id,
                                        n_bc,
                                        same_alt,
                                        error,
                                        D_id,
                                        Me,
                                        ke,
                                        method,
                                        config,
                                        Me_id,
                                        n_te,
                                        Ate_id,
                                    )
                                    futures[task] = thread_pool.submit(
                                        task_thread,
                                        task,
                                        {},
                                        task_queue,
                                        [
                                            futures[ATestTask(m, n_te, Ate_id)],
                                            futures[task_Me],
                                        ],
                                        dir,
                                    )

    wait_exception_mapping(futures)
