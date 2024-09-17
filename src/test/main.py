from itertools import combinations

from scipy.stats import kendalltau

# from ..aggregator import agg_float
from ..model import GroupModel, Model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.fitness import fitness_outranking


def test(A: NormalPerformanceTable, Mo: Model, Me: Model):
    NB_DM = Mo.group_size if isinstance(Mo, GroupModel) else 1
    Ro = (
        [Mo[dm_id].rank(A) for dm_id in range(Mo.group_size)]
        if isinstance(Mo, GroupModel)
        else [Mo.rank(A)]
    )
    Re = (
        [Me[dm_id].rank(A) for dm_id in range(Me.group_size)]
        if isinstance(Me, GroupModel)
        else [Me.rank(A)]
    )

    test_fitness = [fitness_outranking(Ro[dm_id], Re[dm_id]) for dm_id in range(NB_DM)]
    kendall_tau = [
        kendalltau(Ro[dm_id].data, Re[dm_id].data).statistic for dm_id in range(NB_DM)
    ]
    Mo_intra_kendall_tau = (
        [
            kendalltau(Ro[dm_a].data, Ro[dm_b].data).statistic
            for dm_a, dm_b in combinations(range(NB_DM), 2)
        ]
        if NB_DM > 1
        else None
    )
    Me_intra_kendall_tau = (
        [
            kendalltau(Re[dm_a].data, Re[dm_b].data).statistic
            for dm_a, dm_b in combinations(range(NB_DM), 2)
        ]
        if NB_DM > 1
        else None
    )

    return (test_fitness, kendall_tau, Mo_intra_kendall_tau, Me_intra_kendall_tau)
