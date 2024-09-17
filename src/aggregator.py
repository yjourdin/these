from collections.abc import Iterable
from functools import partial

from mcda.internal.core.values import Ranking
from numpy import mean
from pandas import DataFrame

agg_float_func = min
agg_rank_func = partial(mean, axis=0)


def agg_float(data: Iterable[float], **kwargs) -> float:
    return agg_float_func((x for x in data), **kwargs)


def agg_rank(data: Iterable[Ranking], **kwargs) -> Ranking:
    df = DataFrame([r.data for r in data])
    return Ranking(agg_rank_func(df, **kwargs))
