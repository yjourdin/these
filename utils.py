from numpy import sort

from mcda_local.core.performance_table import NormalPerformanceTable, PerformanceTable


def midpoints(performance_table: PerformanceTable) -> PerformanceTable:
    # sort performance table
    df = performance_table.normalize().data.transform(sort)

    # compute midpoints
    df = df.rolling(2).mean().drop(df.index[0])

    # Add 0's and 1's at the beginning and the end
    df.loc[0, :] = [0] * len(performance_table.criteria)
    df.loc[len(performance_table.data), :] = [1] * len(performance_table.criteria)
    df.sort_index(inplace=True)

    result = NormalPerformanceTable(
        df,
        criteria=performance_table.criteria,
    )
    return result.transform(performance_table.scales)


def print_list(lst):
    result = ""
    for x in lst:
        result += f"{x:3.3f} "
    return result
