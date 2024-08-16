from collections import defaultdict
from itertools import product
from typing import cast

from ..methods import MethodEnum
from ..models import ModelEnum
from .arguments import Arguments
from .config import MIPConfig, SAConfig
from .task import ATestTask, ATrainTask, DTask, MIPTask, MoTask, SATask, Task, TestTask


def task_precedence(args: Arguments):
    succeed: defaultdict[Task, list[Task]] = defaultdict(list)
    precede: defaultdict[Task, list[Task]] = defaultdict(list)
    to_do: list[Task] = []

    for m in args.M:
        for n_tr, Atr_id in product(args.N_tr, range(len(args.seeds.A_tr))):
            to_do.append(ATrainTask(args.seeds, m, n_tr, Atr_id))

        for n_te, Ate_id in product(args.N_te, range(len(args.seeds.A_te))):
            to_do.append(ATestTask(args.seeds, m, n_te, Ate_id))

        for Mo, ko, group_size in product(args.Mo, args.Ko, args.group_size):
            for Mo_id in range(len(args.seeds.Mo[group_size])):
                to_do.append(MoTask(args.seeds, m, Mo, ko, group_size, Mo_id))

        for n_tr, Atr_id in product(args.N_tr, range(args.nb_A_tr)):
            t_A_train = ATrainTask(args.seeds, m, n_tr, Atr_id)

            for Mo, ko, group_size in product(
                args.Mo,
                args.Ko,
                args.group_size,
            ):
                for Mo_id in range(len(args.seeds.Mo[group_size])):
                    t_Mo = MoTask(args.seeds, m, Mo, ko, group_size, Mo_id)

                    for n_bc, e, D_id in product(
                        args.N_bc,
                        args.error,
                        range(args.nb_D) if args.nb_D else [Mo_id],
                    ):
                        t_Ds = []
                        for dm_id in range(group_size):
                            t_D = DTask(
                                args.seeds,
                                m,
                                n_tr,
                                Atr_id,
                                Mo,
                                ko,
                                group_size,
                                Mo_id,
                                n_bc,
                                e,
                                dm_id,
                                D_id,
                            )

                            succeed[t_A_train] += [t_D]
                            succeed[t_Mo] += [t_D]
                            precede[t_D] += [t_A_train, t_Mo]
                            t_Ds.append(t_D)

                        for Me, ke, method, Me_id in product(
                            args.Me,
                            args.Ke,
                            args.method,
                            range(args.nb_Me) if args.nb_Me else [D_id],
                        ):
                            for config in (
                                config
                                for config in args.config
                                if config.method == method
                            ):
                                t_Me: Task
                                match method:
                                    case MethodEnum.SA:
                                        t_Me = SATask(
                                            args.seeds,
                                            m,
                                            n_tr,
                                            Atr_id,
                                            Mo,
                                            ko,
                                            group_size,
                                            Mo_id,
                                            n_bc,
                                            e,
                                            D_id,
                                            Me,
                                            ke,
                                            cast(SAConfig, config),
                                            Me_id,
                                        )
                                    case MethodEnum.MIP if Me.value[
                                        0
                                    ] == ModelEnum.SRMP:
                                        t_Me = MIPTask(
                                            args.seeds,
                                            m,
                                            n_tr,
                                            Atr_id,
                                            Mo,
                                            ko,
                                            group_size,
                                            Mo_id,
                                            n_bc,
                                            e,
                                            D_id,
                                            Me,
                                            ke,
                                            cast(MIPConfig, config),
                                            Me_id,
                                        )
                                    case _:
                                        break

                                for t_D in t_Ds:
                                    succeed[t_D] += [t_Me]
                                    precede[t_Me] += [t_D]

                                for n_te, Ate_id in product(
                                    args.N_te,
                                    range(args.nb_A_te) if args.nb_A_te else [Me_id],
                                ):
                                    t_A_test = ATestTask(args.seeds, m, n_te, Ate_id)
                                    t_test = TestTask(
                                        args.seeds,
                                        m,
                                        n_tr,
                                        Atr_id,
                                        Mo,
                                        ko,
                                        group_size,
                                        Mo_id,
                                        n_bc,
                                        e,
                                        D_id,
                                        Me,
                                        ke,
                                        method,
                                        config.id,
                                        Me_id,
                                        n_te,
                                        Ate_id,
                                    )

                                    succeed[t_Me] += [t_test]
                                    succeed[t_A_test] += [t_test]
                                    precede[t_test] += [t_A_test, t_Me]

    return to_do, succeed, precede
