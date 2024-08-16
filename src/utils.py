from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, TypeVar

import numpy as np
from mcda.internal.core.scales import OrdinalScale
from mcda.matrices import PerformanceTable

S = TypeVar("S", bound=OrdinalScale, covariant=True)


def midpoints(performance_table: PerformanceTable[S]) -> PerformanceTable[S]:
    # sort performance table
    df = performance_table.data.transform(np.sort)

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


def print_list(lst: list):
    result = ""
    for x in lst:
        result += f"{x:1.3f} "
    return result


def to_str(o):
    match o:
        case list():
            return "[" + ", ".join([to_str(i) for i in o]) + "]"
        case Enum():
            return to_str(o.name)
        case _:
            return str(o).title().replace("_", "")


def filename(dct: dict[str, Any], ext: str):
    return (
        "_".join(to_str(k) + "_" + to_str(v) for k, v in dct.items() if k != "self")
        + "."
        + ext
    )


filename_csv = partial(filename, ext="csv")
filename_json = partial(filename, ext="json")


def max_weight(n):
    # return 1 if n == 1 else (n - 1) * max_weight(n - 1) + 1
    return 2**n


def compose(*fs: Callable):
    def compose2(f: Callable, g: Callable):
        return lambda *a, **kw: g(f(*a, **kw))

    return reduce(compose2, fs) if fs else None
