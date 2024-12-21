from collections.abc import Iterator
from enum import Enum
from functools import partial, reduce
from itertools import chain, count
from typing import Any, Callable, NamedTuple

import numpy as np
from mcda import PerformanceTable


def midpoints(performance_table: PerformanceTable) -> PerformanceTable:
    # sort performance table
    df = performance_table.data.transform(np.sort)

    # compute midpoints
    df = df.rolling(2).mean().drop(df.index[0])

    # add 0's and 1's at the beginning and the end
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
        if isinstance(x, str):
            s = x
        else:
            s = f"{x:1.3f} "
        result += s
    return result


def to_str(o):
    match o:
        case list():
            return "[" + ", ".join([to_str(i) for i in o]) + "]"
        case Enum():
            return str(o)
        case _:
            return str(o).title().replace("_", "")


def dict_values_to_str(dct: dict):
    return {str(k): str(v) for k, v in dct.items()}


class Cell(NamedTuple):
    name: str
    value: Any


def add_str_to_list(
    o, index: Iterator[int] | None = None, prefix: list[str] = []
) -> list[Cell]:
    index = index or count()
    if np.ndim(o) > 0 and len(o) == 1:
        o = o[0]
    if np.ndim(o) == 0:
        return [Cell("_".join(prefix), o)]
    else:
        return list(
            chain.from_iterable(
                add_str_to_list(
                    val, (i for i in range(len(o)) if i != ind), prefix + [str(ind)]
                )
                for ind, val in zip(index, o)
            )
        )


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
    return 2**n - 1


def compose(*fs: Callable):
    def compose2(f: Callable, g: Callable):
        return lambda *a, **kw: g(f(*a, **kw))

    return reduce(compose2, fs)

def list_replace(a: list[Any], b: list[Any]):
    a[:len(b)] = b[:len(a)]

