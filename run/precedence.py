from collections import defaultdict
from itertools import product
from typing import cast

from .arguments import Arguments
from .config import MIPConfig, SAConfig
from .task import ATestTask, ATrainTask, DTask, MIPTask, MoTask, SATask, Task, TestTask


def task_precedence(args: Arguments):
    succeed: defaultdict[Task, list[Task]] = defaultdict(list)
    precede: defaultdict[Task, list[Task]] = defaultdict(list)
    to_do: list[Task] = []

    for m in args.M:
        for n_tr, A_tr_id in product(args.N_tr, range(len(args.seeds.A_train))):
            to_do.append(ATrainTask(m, n_tr, A_tr_id))  # Put task in task queue

        for n_te, A_te_id in product(args.N_te, range(len(args.seeds.A_test))):
            to_do.append(ATestTask(m, n_te, A_te_id))  # Put task in task queue

        for Mo in args.Mo:
            for ko, Mo_id in product(args.Ko, range(len(args.seeds.Mo))):
                to_do.append(MoTask(m, Mo, ko, Mo_id))  # Put task in task queue

        for n_tr, A_tr_id in product(args.N_tr, range(args.nb_A_tr)):
            t_A_train = ATrainTask(m, n_tr, A_tr_id)

            for Mo, ko, Mo_id in product(
                args.Mo, args.Ko, range(args.nb_Mo) if args.nb_Mo else [A_tr_id]
            ):
                t_Mo = MoTask(m, Mo, ko, Mo_id)

                for n_bc, e in product(args.N_bc, args.error):
                    t_D = DTask(m, n_tr, A_tr_id, Mo, ko, Mo_id, n_bc, e)

                    succeed[t_A_train] += [t_D]
                    succeed[t_Mo] += [t_D]
                    precede[t_D] += [t_A_train, t_Mo]

                    for Me, ke, method in product(args.Me, args.Ke, args.method):
                        for config in (
                            config for config in args.config if config.method == method
                        ):
                            t_Me: Task
                            match method:
                                case "SA":
                                    t_Me = SATask(
                                        m,
                                        n_tr,
                                        A_tr_id,
                                        Mo,
                                        ko,
                                        Mo_id,
                                        n_bc,
                                        e,
                                        Me,
                                        ke,
                                        cast(SAConfig, config),
                                    )
                                case "MIP" if Me == "SRMP":
                                    t_Me = MIPTask(
                                        m,
                                        n_tr,
                                        A_tr_id,
                                        Mo,
                                        ko,
                                        Mo_id,
                                        n_bc,
                                        e,
                                        ke,
                                        cast(MIPConfig, config),
                                    )
                                case _:
                                    break

                            succeed[t_D] += [t_Me]
                            precede[t_Me] += [t_D]

                            for n_te, A_te_id in product(
                                args.N_te,
                                range(args.nb_A_te) if args.nb_A_te else [Mo_id],
                            ):
                                t_A_test = ATestTask(m, n_te, A_te_id)
                                t_test = TestTask(
                                    m,
                                    n_tr,
                                    A_tr_id,
                                    Mo,
                                    ko,
                                    Mo_id,
                                    n_bc,
                                    e,
                                    Me,
                                    ke,
                                    method,
                                    config.id,
                                    n_te,
                                    A_te_id,
                                )

                                succeed[t_Me] += [t_test]
                                succeed[t_A_test] += [t_test]
                                precede[t_test] += [t_A_test, t_Me]

    return to_do, succeed, precede
