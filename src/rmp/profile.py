import numpy as np
from pandas import DataFrame

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..random import RNGParam, rng_


def random_profiles(
    nb: int,
    nb_crit: int,
    rng: RNGParam = None,
    profiles_values: NormalPerformanceTable | None = None,
):
    rng = rng_(rng)
    if profiles_values:
        idx = np.sort(rng.choice(len(profiles_values.data), (nb, nb_crit)))
        return NormalPerformanceTable(
            DataFrame({
                i: profiles_values.data.iloc[idx[:, i], i].to_list()
                for i in range(nb_crit)
            })
        )
    else:
        return NormalPerformanceTable(np.sort(rng.random((nb, nb_crit)), 0))
