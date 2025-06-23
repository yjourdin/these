from typing import Any

import pandas as pd
from mcda import PerformanceTable
from mcda.internal.core.scales import NormalScale, QuantitativeScale
from mcda.outranking.srmp import SRMP

from ..constants import DECIMALS
from ..random import RNGParam, rng_


class NormalPerformanceTable(PerformanceTable[NormalScale]):
    def __init__(self, data: Any, *args: Any, **kwargs: Any):
        super().__init__(
            pd.DataFrame(data).round(DECIMALS),
            scales=QuantitativeScale.normal(),
            **kwargs,
        )

    @classmethod
    def random(cls, nb_alt: int, nb_crit: int, rng: RNGParam = None):
        return cls(rng_(rng).random((nb_alt, nb_crit)).round(DECIMALS))

    def plot(self, *args: Any, **kwargs: Any):
        return SRMP.plot_input_data(self, *args, **kwargs)
