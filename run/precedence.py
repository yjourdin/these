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

    nb_DM = args.seeds if isinstance(args.seeds, int) else len(args.seeds)

    for i in range(nb_DM):
        for m in args.M:
            for n_te in args.N_te:
                t_A_test = task_A_test(i, n_te, m)
                to_do.append(t_A_test)  # Put task in task queue
        for n_tr in args.N_tr:
            t_A_train = task_A_train(i, n_tr, m)
            to_do.append(t_A_train)  # Put task in task queue
            for Mo in args.Mo:
                for ko in args.Ko:
                    t_Mo = task_Mo(i, m, Mo, ko)
                    to_do.append(t_Mo)  # Put task in task queue
                    for n_bc in args.N_bc:
                        for e in args.error:
                            t_D = task_D(i, n_tr, m, Mo, ko, n_bc, e)
                            succeed[t_A_train] += [t_D]
                            succeed[t_Mo] += [t_D]
                            precede[t_D] += [t_A_train, t_Mo]
                            for Me in args.Me:
                                if Me == Mo:
                                    for ke in args.Ke:
                                        ts = []
                                        for method in args.method:
                                            match method:
                                                case "SA":
                                                    for config in args.config:
                                                        t_Me = task_SA(
                                                            i,
                                                            n_tr,
                                                            m,
                                                            Mo,
                                                            ko,
                                                            n_bc,
                                                            e,
                                                            Me,
                                                            ke,
                                                            config,
                                                        )
                                                        t_test = task_test(
                                                            i,
                                                            n_tr,
                                                            n_te,
                                                            m,
                                                            Mo,
                                                            ko,
                                                            n_bc,
                                                            e,
                                                            Me,
                                                            ke,
                                                            method,
                                                            config,
                                                        )
                                                        ts.append((t_Me, t_test))
                                                case "MIP" if Me == "SRMP":
                                                    t_Me = task_MIP(
                                                        i, n_tr, m, Mo, ko, n_bc, e, ke
                                                    )
                                                    t_test = task_test(
                                                        i,
                                                        n_tr,
                                                        n_te,
                                                        m,
                                                        Mo,
                                                        ko,
                                                        n_bc,
                                                        e,
                                                        Me,
                                                        ke,
                                                        method,
                                                    )
                                                    ts.append((t_Me, t_test))
                                                case _:
                                                    break
                                            for n_te in args.N_te:
                                                for t_Me, t_test in ts:
                                                    succeed[t_D] += [t_Me]
                                                    precede[t_Me] += [t_D]
                                                    succeed[t_Me] += [t_test]
                                                    succeed[t_A_test] += [t_test]
                                                    precede[t_test] += [t_A_test, t_Me]

    return to_do, succeed, precede