from typing import cast

from numpy import sort

from mcda_local.core.performance_table import PerformanceTable
from mcda_local.learner.neighbor import Neighbor
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP


class NeighborProfiles(Neighbor[RMP | SRMP]):
    def __init__(self, values: PerformanceTable):
        self.values = values

    def __call__(self, model, rng):
        neighbor = model.copy()

        crit_ind = rng.choice(len(neighbor.profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        value = neighbor.profiles.data.iloc[profile_ind, crit_ind]
        value_ind = cast(int, crit_values[crit_values == value].index[0])

        match 0 < value_ind, value_ind < (len(self.values.alternatives) - 1):
            case True, True:
                new_value_ind = rng.choice([value_ind - 1, value_ind + 1])
                neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                    new_value_ind
                ]
            case True, False:
                neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                    value_ind - 1
                ]
            case False, True:
                neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                    value_ind + 1
                ]

        neighbor.profiles.data.transform(sort)

        return neighbor


class NeighborWeights(Neighbor[SRMP]):
    def __init__(self, amp):
        self.amp = amp

    def __call__(self, model, rng):
        neighbor = model.copy()

        d = neighbor.criteria_weights
        keys = list(d)
        crit = rng.choice(keys)

        x = rng.uniform(-1, 1)

        neighbor.criteria_weights[crit] = min(max(d[crit] + self.amp * x, 0), 1)

        return neighbor


class NeighborCapacities(Neighbor[RMP]):
    def __call__(self, model, rng):
        neighbor = model.copy()
        power_set = neighbor.criteria_capacities
        keys = list(power_set)
        crits = keys[rng.choice(len(keys))]
        while power_set.min_capacity(crits) == power_set.max_capacity(crits):
            crits = keys[rng.choice(len(keys))]
        capacity = power_set[crits]
        if power_set.min_capacity(crits) < capacity < power_set.max_capacity(crits):
            new_capa = rng.choice([capacity - 1, capacity + 1])
        elif power_set.min_capacity(crits) == capacity:
            new_capa = capacity + 1
        elif capacity == power_set.max_capacity(crits):
            new_capa = capacity - 1
        else:
            raise ValueError(
                f"{capacity} is not between {power_set.min_capacity(crits)}"
                f" and {power_set.max_capacity(crits)}"
            )
        neighbor.criteria_capacities[crits] = new_capa
        return neighbor


class NeighborLexOrder(Neighbor[RMP | SRMP]):
    def __call__(self, model, rng):
        neighbor = model.copy()
        lex_order = neighbor.lexicographic_order
        i = rng.choice(len(lex_order) - 1)
        neighbor.lexicographic_order[i], neighbor.lexicographic_order[i + 1] = (
            lex_order[i + 1],
            lex_order[i],
        )
        return neighbor
