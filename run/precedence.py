from collections import defaultdict

from .arguments import Arguments
from .task import (
    Task,
    task_A_test,
    task_A_train,
    task_D,
    task_MIP,
    task_Mo,
    task_SA,
    task_test,
)


def task_precedence(args: Arguments):
    succeed: defaultdict[Task, list[Task]] = defaultdict(list)
    precede: defaultdict[Task, list[Task]] = defaultdict(list)
    to_do: list[Task] = []

    nb_A_tr = (
        args.A_tr_seeds if isinstance(args.A_tr_seeds, int) else len(args.A_tr_seeds)
    )
    nb_Mo = args.Mo_seeds if isinstance(args.Mo_seeds, int) else len(args.Mo_seeds)
    nb_A_te = (
        args.A_te_seeds if isinstance(args.A_te_seeds, int) else len(args.A_te_seeds)
    )

    for m in args.M:
        for n_tr in args.N_tr:
            for A_tr_id in range(nb_A_tr):
                to_do.append(task_A_train(m, n_tr, A_tr_id))  # Put task in task queue

        for n_te in args.N_te:
            for A_te_id in range(nb_A_te):
                to_do.append(task_A_test(m, n_te, A_te_id))  # Put task in task queue

        for Mo in args.Mo:
            for ko in args.Ko:
                for Mo_id in range(nb_Mo):
                    to_do.append(task_Mo(m, Mo, ko, Mo_id))  # Put task in task queue

        for n_tr in args.N_tr:
            for A_tr_id in range(nb_A_tr):
                t_A_train = task_A_train(m, n_tr, A_tr_id)

                for Mo in args.Mo:
                    for ko in args.Ko:
                        for Mo_id in range(nb_Mo):
                            t_Mo = task_Mo(m, Mo, ko, Mo_id)

                            for n_bc in args.N_bc:
                                for e in args.error:
                                    t_D = task_D(
                                        m, n_tr, A_tr_id, Mo, ko, Mo_id, n_bc, e
                                    )

                                    succeed[t_A_train] += [t_D]
                                    succeed[t_Mo] += [t_D]
                                    precede[t_D] += [t_A_train, t_Mo]

                                    for Me in args.Me:
                                        if Me == Mo:
                                            for ke in args.Ke:
                                                for method in args.method:
                                                    for config in args.config[method]:
                                                        match method:
                                                            case "SA":
                                                                t_Me = task_SA(
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
                                                                    config,
                                                                )

                                                            case "MIP" if Me == "SRMP":
                                                                t_Me = task_MIP(
                                                                    m,
                                                                    n_tr,
                                                                    A_tr_id,
                                                                    Mo,
                                                                    ko,
                                                                    Mo_id,
                                                                    n_bc,
                                                                    e,
                                                                    ke,
                                                                )

                                                            case _:
                                                                break

                                                        succeed[t_D] += [t_Me]
                                                        precede[t_Me] += [t_D]

                                                        for n_te in args.N_te:
                                                            for A_te_id in range(
                                                                nb_A_te
                                                            ):
                                                                t_A_test = task_A_test(
                                                                    m, n_te, A_te_id
                                                                )
                                                                t_test = task_test(
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
                                                                    config,
                                                                    n_te,
                                                                    A_te_id,
                                                                )

                                                                succeed[t_Me] += [
                                                                    t_test
                                                                ]
                                                                succeed[t_A_test] += [
                                                                    t_test
                                                                ]
                                                                precede[t_test] += [
                                                                    t_A_test,
                                                                    t_Me,
                                                                ]

    return to_do, succeed, precede
