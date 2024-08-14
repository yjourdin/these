from numpy.random import Generator
from pandas import read_csv
from scipy.stats import kendalltau

from ..methods import MethodEnum
from ..mip.main import learn_mip
from ..model import Group
from ..models import GroupModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.fitness import fitness_outranking
from ..preference_structure.generate import noisy_comparisons, random_comparisons
from ..preference_structure.io import from_csv, to_csv
from ..rmp.model import RMPModel
from ..sa.main import learn_sa
from ..srmp.model import SRMPModel, SRMPParamEnum
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
    group_id: int,
    dm_id: int,
    dir: Directory,
    rng: Generator,
):
    match model:
        case ModelEnum.RMP:
            Mo = RMPModel.random(k, m, rng)
        case ModelEnum.SRMP:
            Mo = SRMPModel.random(k, m, rng)
    with dir.Mo(m, model, k, group_size, group_id, dm_id).open("w") as f:
        f.write(Mo.to_json())


def create_D(
    m: int,
    ntr: int,
    Atr_id: int,
    Mo: GroupModelEnum,
    ko: int,
    group_size: int,
    group_id: int,
    dm_id: int,
    n: int,
    error: float,
    id: int,
    dir: Directory,
    rng: Generator,
):
    with dir.Mo(m, Mo, ko, group_size, group_id, dm_id).open("r") as f:
        match Mo:
            case ModelEnum.RMP:
                model = RMPModel.from_json(f.read())
            case ModelEnum.SRMP:
                model = SRMPModel.from_json(f.read())

    with dir.A_train(m, ntr, Atr_id).open("r") as f:
        A = NormalPerformanceTable(read_csv(f, header=None))

    D = random_comparisons(n, A, model, rng)

    if error:
        D = noisy_comparisons(D, error, rng)

    with dir.D(m, ntr, Atr_id, Mo, ko, group_size, group_id, dm_id, n, error, id).open(
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
    group_id: int,
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
        with dir.D(m, ntr, Atr_id, Mo, ko, group_size, group_id, dm_id, n, e, id).open(
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
            group_id,
            n,
            e,
            D_id,
            Me,
            shared_params,
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
    group_id: int,
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
        with dir.D(m, ntr, Atr_id, Mo, ko, group_size, group_id, dm_id, n, e, id).open(
            "r"
        ) as f:
            D.append(from_csv(f))

    rng_init, rng_sa = rng.spawn(2)
    best_model, best_fitness, time, it = learn_sa(
        Me,
        ke,
        A,
        D[0],
        config.alpha,
        rng_init,
        rng_sa,
        accept=config.accept,
        max_it=config.max_it,
        **(
            {"amp": config.amp} if isinstance(config, SRMPSAConfig) else {}
        ),  # type: ignore
    )

    with dir.Me(
        m,
        ntr,
        Atr_id,
        Mo,
        ko,
        1,
        group_id,
        n,
        e,
        D_id,
        Me,
        shared_params,
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
    group_id: int,
    n: int,
    e: float,
    D_id: int,
    Me_type: GroupModelEnum,
    shared_params: list[SRMPParamEnum],
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

    Mo = Group()
    Me = Group()
    for dm_id in range(group_size):
        with dir.Mo(m, Mo_type, ko, group_size, group_id, dm_id).open("r") as f:
            match Mo_type:
                case ModelEnum.RMP:
                    Mo.append(RMPModel.from_json(f.read()))
                case ModelEnum.SRMP:
                    Mo.append(SRMPModel.from_json(f.read()))

        with dir.Me(
            m,
            ntr,
            Atr_id,
            Mo_type,
            ko,
            group_size,
            group_id,
            n,
            e,
            D_id,
            Me_type,
            shared_params,
            ke,
            method,
            config_id,
            Me_id,
        ).open("r") as f:
            match Me_type:
                case ModelEnum.RMP:
                    Me = RMPModel.from_json(f.read())
                case ModelEnum.SRMP:
                    Me = SRMPModel.from_json(f.read())

    Ro = Mo.rank(A_test)
    Re = Me.rank(A_test)

    test_fitness = group_agg(zip(Ro, Re), lambda x: fitness_outranking(x[0], x[1]))
    kendall_tau = group_agg(
        zip(Ro, Re), lambda x: kendalltau(x[0].data, x[1].data).statistic
    )

    return (test_fitness, kendall_tau)
