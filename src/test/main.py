from scipy.stats import kendalltau

from ..aggregator import agg_float
from ..model import GroupModel, Model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.fitness import fitness_outranking


def test(A: NormalPerformanceTable, Mo: Model, Me: Model):
    NB_DM = Mo.size if isinstance(Mo, GroupModel) else 1
    Ro = (
        [Mo[dm_id].rank(A) for dm_id in range(Mo.size)]
        if isinstance(Mo, GroupModel)
        else [Mo.rank(A)]
    )
    Re = (
        [Me[dm_id].rank(A) for dm_id in range(Me.size)]
        if isinstance(Me, GroupModel)
        else [Me.rank(A)]
    )

    test_fitness = agg_float(
        fitness_outranking(Ro[dm_id], Re[dm_id]) for dm_id in range(NB_DM)
    )
    kendall_tau = agg_float(
        kendalltau(Ro[dm_id].data, Re[dm_id].data).statistic for dm_id in range(NB_DM)
    )

    return (test_fitness, kendall_tau)
