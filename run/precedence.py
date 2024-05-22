from collections import defaultdict

from .arguments import Arguments
from .task import ATestTask, ATrainTask, DTask, MIPTask, MoTask, SATask, Task, TestTask


def task_precedence(args: Arguments):
    succeed: defaultdict[Task, list[Task]] = defaultdict(list)
    precede: defaultdict[Task, list[Task]] = defaultdict(list)
    to_do: list[Task] = []

    nb_A_tr = (
        args.seeds.A_train
        if isinstance(args.seeds.A_train, int)
        else len(args.seeds.A_train)
    )
    nb_Mo = args.seeds.Mo if isinstance(args.seeds.Mo, int) else len(args.seeds.Mo)
    nb_A_te = (
        args.seeds.A_test
        if isinstance(args.seeds.A_test, int)
        else len(args.seeds.A_test)
    )

    for m in args.M:
        for n_tr in args.N_tr:
            for A_tr_id in range(nb_A_tr):
                to_do.append(ATrainTask(m, n_tr, A_tr_id))  # Put task in task queue

        for n_te in args.N_te:
            for A_te_id in range(nb_A_te):
                to_do.append(ATestTask(m, n_te, A_te_id))  # Put task in task queue

        for Mo in args.Mo:
            for ko in args.Ko:
                for Mo_id in range(nb_Mo):
                    to_do.append(MoTask(m, Mo, ko, Mo_id))  # Put task in task queue

        for n_tr in args.N_tr:
            for A_tr_id in range(nb_A_tr):
                t_A_train = ATrainTask(m, n_tr, A_tr_id)

                for Mo in args.Mo:
                    for ko in args.Ko:
                        for Mo_id in range(nb_Mo):
                            t_Mo = MoTask(m, Mo, ko, Mo_id)

                            for n_bc in args.N_bc:
                                for e in args.error:
                                    t_D = DTask(
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
                                                                    config,
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
                                                                )
                                                            case _:
                                                                break

                                                        succeed[t_D] += [t_Me]
                                                        precede[t_Me] += [t_D]

                                                        for n_te in args.N_te:
                                                            for A_te_id in range(
                                                                nb_A_te
                                                            ):
                                                                t_A_test = ATestTask(
                                                                    m, n_te, A_te_id
                                                                )
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
