from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

from ..aggregator import agg_float
from ..methods import MethodEnum
from ..mip.main import learn_mip
from ..models import GroupModelEnum, group_model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.fitness import fitness_outranking
from ..preference_structure.generate import noisy_comparisons, random_comparisons
from ..preference_structure.io import from_csv, to_csv
from ..sa.main import learn_sa
from .config import MIPConfig, SAConfig, SRMPSAConfig
from .directory import Directory


def create_A_train(m: int, n: int, id: int, dir: Directory, rng: Generator):
    A = NormalPerformanceTable.random(n, m, rng)
    with dir.A_train(m, n, id).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_A_test(m: int, n: int, id: int, dir: Directory, rng: Generator):
    A = NormalPerformanceTable.random(n, m, rng)
    with dir.A_test(m, n, id).open("w") as f:
        A.data.to_csv(f, header=False, index=False)


def create_Mo(
    m: int,
    model: GroupModelEnum,
    k: int,
    group_size: int,
    id: int,
    dir: Directory,
    rng: Generator,
):
    Mo = group_model(*model.value).random(
        size=group_size, nb_profiles=k, nb_crit=m, rng=rng
    )
    with dir.Mo(m, model, k, group_size, id).open("w") as f:
        f.write(Mo.to_json())


def create_D(
    m: int,
    ntr: int,
    Atr_id: int,
    Mo: GroupModelEnum,
    ko: int,
    group_size: int,
    Mo_id: int,
    n: int,
    error: float,
    dm_id: int,
    id: int,
    dir: Directory,
    rng: Generator,
):
    with dir.Mo(m, Mo, ko, group_size, Mo_id).open("r") as f:
        model = group_model(*Mo.value).from_json(f.read())

    with dir.A_train(m, ntr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = random_comparisons(n, A, model[dm_id], rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D(m, ntr, Atr_id, Mo, ko, group_size, Mo_id, n, error, dm_id, id).open(
        "w"
    ) as f:
        to_csv(D, f)


def run_MIP(
    m: int,
    ntr: int,
    Atr_id: int,
    Mo: GroupModelEnum,
    ko: int,
    group_size: int,
    Mo_id: int,
    n: int,
    e: float,
    D_id: int,
    Me: GroupModelEnum,
    ke: int,
    config: MIPConfig,
    id: int,
    dir: Directory,
    seed: int,
):
    with dir.A_train(m, ntr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = []
    for dm_id in range(group_size):
        with dir.D(m, ntr, Atr_id, Mo, ko, group_size, Mo_id, n, e, dm_id, id).open(
            "r"
        ) as f:
            D.append(from_csv(f))

    best_model, best_fitness, time = learn_mip(
        Me, ke, A, D, seed=seed, gamma=config.gamma
    )
    if best_model:
        with dir.Me(
            m,
            ntr,
            Atr_id,
            Mo,
            ko,
            group_size,
            Mo_id,
            n,
            e,
            D_id,
            Me,
            ke,
            MethodEnum.MIP,
            config.id,
            id,
        ).open("w") as f:
            f.write(best_model.to_json())

    return (time, best_fitness)


def run_SA(
    m: int,
    ntr: int,
    Atr_id: int,
    Mo: GroupModelEnum,
    ko: int,
    group_size: int,
    Mo_id: int,
    n: int,
    e: float,
    D_id: int,
    Me: GroupModelEnum,
    ke: int,
    config: SAConfig,
    id: int,
    dir: Directory,
    rng: Generator,
):
    with dir.A_train(m, ntr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = []
    for dm_id in range(group_size):
        with dir.D(m, ntr, Atr_id, Mo, ko, group_size, Mo_id, n, e, dm_id, id).open(
            "r"
        ) as f:
            D.append(from_csv(f))

    rng_init, rng_sa = rng.spawn(2)
    best_model, best_fitness, time, it = learn_sa(
        Me.value[0],
        ke,
        A,
        D[0],
        config.alpha,
        rng_init,
        rng_sa,
        accept=config.accept,
        max_it=config.max_it,
        **(
            {"amp": config.amp}
            if isinstance(config, SRMPSAConfig)
            else {}  # type: ignore
        ),
    )

    with dir.Me(
        m,
        ntr,
        Atr_id,
        Mo,
        ko,
        1,
        Mo_id,
        n,
        e,
        D_id,
        Me,
        ke,
        MethodEnum.SA,
        config.id,
        id,
    ).open("w") as f:
        f.write(best_model.to_json())

    return (time, it, best_fitness)


def run_test(
    m: int,
    ntr: int,
    Atr_id: int,
    Mo_type: GroupModelEnum,
    ko: int,
    group_size: int,
    Mo_id: int,
    n: int,
    e: float,
    D_id: int,
    Me_type: GroupModelEnum,
    ke: int,
    method: MethodEnum,
    config_id: int,
    Me_id: int,
    nte: int,
    Ate_id: int,
    dir: Directory,
):
    with dir.A_test(m, nte, Ate_id).open("r") as f:
        A_test = NormalPerformanceTable(read_csv(f, header=None))

    with dir.Mo(m, Mo_type, ko, group_size, Mo_id).open("r") as f:
        Mo = group_model(*Mo_type.value).from_json(f.read())

        with dir.Me(
            m,
            ntr,
            Atr_id,
            Mo_type,
            ko,
            group_size,
            Mo_id,
            n,
            e,
            D_id,
            Me_type,
            ke,
            method,
            config_id,
            Me_id,
        ).open("r") as f:
            Me = group_model(*Me_type.value).from_json(f.read())

    Ro = [Mo[dm_id].rank(A_test) for dm_id in range(group_size)]
    Re = [Me[dm_id].rank(A_test) for dm_id in range(group_size)]

    test_fitness = agg_float(
        fitness_outranking(Ro[dm_id], Re[dm_id]) for dm_id in range(group_size)
    )
    kendall_tau = agg_float(
        kendalltau(Ro[dm_id].data, Re[dm_id].data).statistic
        for dm_id in range(group_size)
    )

    return (test_fitness, kendall_tau)
