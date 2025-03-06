from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import mean

agg_float_func = min
agg_rank_func = partial(mean, axis=0)


def agg_float(data: Iterable[float], **kwargs: Any) -> float:
    return agg_float_func((x for x in data), **kwargs)


def agg_rank(
    data: Iterable[npt.NDArray[np.int_]], **kwargs: Any
) -> npt.NDArray[np.int_]:
    return agg_rank_func(np.stack(list(data)))
