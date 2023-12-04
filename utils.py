from numpy import sort

from mcda_local.core.performance_table import NormalPerformanceTable, PerformanceTable
from mcda_local.core.values import Ranking


def midpoints(performance_table: PerformanceTable) -> PerformanceTable:
    df = performance_table.normalize().data.transform(sort)
    df = df.rolling(2).mean().drop(df.index[0])

    df.loc[0, :] = [0] * len(performance_table.criteria)
    df.loc[len(performance_table.data), :] = [1] * len(performance_table.criteria)

    df.sort_index(inplace=True)

    result = NormalPerformanceTable(
        df,
        criteria=performance_table.criteria,
    )
    return result.transform(performance_table.scales)


def kendall_tau(rank1: Ranking, rank2: Ranking) -> float:
    pass
