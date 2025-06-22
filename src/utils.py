from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import Enum
from functools import partial, reduce
from itertools import chain, count
from pathlib import Path
from typing import Any, NamedTuple, overload

import numpy as np
import numpy.typing as npt
from mcda import PerformanceTable

from .constants import EPSILON


def midpoints(performance_table: PerformanceTable[Any]) -> PerformanceTable[Any]:
    # sort performance table
    df = performance_table.data.transform(np.sort)

    # compute midpoints
    df = df.rolling(2).mean().reset_index(drop=True).drop(0)

    # add 0's and 1's at the beginning and the end
    df.loc[0, :] = [
        performance_table.scales[c].interval.dmin for c in performance_table.criteria
    ]
    df.loc[len(performance_table.data), :] = [
        performance_table.scales[c].interval.dmax for c in performance_table.criteria
    ]
    df.sort_index(inplace=True)

    return PerformanceTable(df, scales=performance_table.scales)


def print_list(lst: Iterable[str | int]):
    result = ""
    for x in lst:
        if isinstance(x, str):
            s = x
        else:
            s = f"{x:1.3f} "
        result += s
    return result


def to_str(o: Any) -> str:
    match o:
        case list():
            o_str = [to_str(i) for i in o]
            return "[" + ", ".join(o_str) + "]"
        case Enum():
            return str(o)
        case _:
            return str(o).title().replace("_", "")


def list_str(lst: Iterable[Any]):
    return [str(x) for x in lst]


def dict_str(dct: Mapping[Any, Any]):
    return {str(k): str(v) for k, v in dct.items()}


class Cell(NamedTuple):
    name: str
    value: Any


def add_str_to_list(
    o: Any, index: Iterator[int] | None = None, prefix: list[str] = []
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


def pathname(dct: dict[str, Any], ext: str):
    return (
        "_".join(to_str(k) + "_" + to_str(v) for k, v in dct.items() if k != "self")
        + ext
    )


dirname = partial(pathname, ext="")
filename_csv = partial(pathname, ext=".csv")
filename_json = partial(pathname, ext=".json")


def max_weight(n: int) -> int:
    # return 1 if n == 1 else (n - 1) * max_weight(n - 1) + 1
    return 2**n - 1


def compose(*fs: Callable[..., Any]):
    def compose2(f: Callable[..., Any], g: Callable[..., Any]):
        def func(*args: Any, **kwargs: Any):
            return g(f(*args, **kwargs))

        return func

    return reduce(compose2, fs)


def list_replace(a: Sequence[Any], b: Sequence[Any]):
    result = list(a)
    result[: len(b)] = b[: len(a)]
    return result


def add_filename_suffix(filename: Path, suffix: str):
    return filename.parent / (filename.stem + suffix + filename.suffix)


def round_epsilon(x: float, epsilon: float = EPSILON) -> float:
    return np.round(x / EPSILON) * EPSILON


@overload
def tolist(a: npt.NDArray[np.float64]) -> list[float]: ...
@overload
def tolist(a: npt.NDArray[np.int_]) -> list[int]: ...


def tolist(a: npt.NDArray[Any]) -> list[Any]:
    return a.tolist() if a.ndim != 0 else [a.tolist()]


def int_to_ind_set(x: int):
    return [i for i in range(x.bit_length()) if (x >> i) & 1]
