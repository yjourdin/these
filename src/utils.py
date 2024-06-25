from typing import TypeVar

from mcda.internal.core.scales import OrdinalScale
from mcda.matrices import PerformanceTable
from numpy import sort

S = TypeVar("S", bound=OrdinalScale, covariant=True)


def midpoints(performance_table: PerformanceTable[S]) -> PerformanceTable[S]:
    # sort performance table
    df = performance_table.data.transform(sort)

    # compute midpoints
    df = df.rolling(2).mean().drop(df.index[0])

    # Add 0's and 1's at the beginning and the end
    df.loc[0, :] = [
        performance_table.scales[c].interval.dmin for c in performance_table.criteria
    ]
    df.loc[len(performance_table.data), :] = [
        performance_table.scales[c].interval.dmax for c in performance_table.criteria
    ]
    df.sort_index(inplace=True)

    return PerformanceTable(df, scales=performance_table.scales)


def print_list(lst):
    result = ""
    for x in lst:
        result += f"{x:1.3f} "
    return result


def max_weight(n):
    # return 1 if n == 1 else (n - 1) * max_weight(n - 1) + 1
    return 2**n
