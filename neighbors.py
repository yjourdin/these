from copy import deepcopy
from typing import cast

from numpy import sort

from mcda_local.core.performance_table import PerformanceTable
from mcda_local.learner.neighbor import Neighbor
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP

# from utils import max_weight


class NeighborProfiles(Neighbor[RMP | SRMP]):
    def __init__(self, values: PerformanceTable):
        self.values = values

    def __call__(self, model, rng):
        neighbor = deepcopy(model)

        # crit_ind = rng.choice(len(neighbor.profiles.criteria))
        # profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        # value = neighbor.profiles.data.iloc[profile_ind, crit_ind]

        # neighbor.profiles.data.iloc[profile_ind, crit_ind] = rng.uniform(
        #     max(value - self.amp, 0), min(value + self.amp, 1)
        # )

        crit_ind = rng.choice(len(neighbor.profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        value = neighbor.profiles.data.iloc[profile_ind, crit_ind]
        value_ind = cast(
            int, crit_values.index.get_loc(crit_values[crit_values == value].index[0])
        )

        # if value_ind == 0:
        #     neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
        #         crit_values.index[value_ind + 1]
        #     ]
        # elif value_ind == (len(self.values.alternatives) - 1):
        #     neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
        #         crit_values.index[value_ind - 1]
        #     ]
        # else:
        #     new_value_ind = rng.choice([value_ind - 1, value_ind + 1])
        #     neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
        #         crit_values.index[new_value_ind]
        #     ]

        new_value_ind = rng.choice(len(self.values.alternatives))
        neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
            crit_values.index[new_value_ind]
        ]

        neighbor.profiles.data.transform(sort)

        return neighbor


class NeighborProfilesSRMP(Neighbor[SRMP]):
    def __init__(self, values: PerformanceTable):
        self.values = values

    def __call__(self, model, rng):
        neighbor = deepcopy(model)

        crit_ind = rng.choice(
            [
                i
                for i, x in enumerate(neighbor.profiles.criteria)
                if neighbor.criteria_weights[x] != 0
            ]
        )
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        value = neighbor.profiles.data.iloc[profile_ind, crit_ind]
        value_ind = cast(int, crit_values[crit_values == value].index[0])

        if value_ind == 0:
            neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                value_ind + 1
            ]
        elif value_ind == (len(self.values.alternatives) - 1):
            neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                value_ind - 1
            ]
        else:
            new_value_ind = rng.choice([value_ind - 1, value_ind + 1])
            neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
                new_value_ind
            ]

        neighbor.profiles.data.transform(sort)

        return neighbor


class NeighborWeights(Neighbor[SRMP]):
    def __init__(self, amp: float):
        self.amp = amp

    def __call__(self, model, rng):
        neighbor = deepcopy(model)

        d = neighbor.criteria_weights
        crit = rng.choice(list(d))

        neighbor.criteria_weights[crit] = rng.uniform(
            max(d[crit] - self.amp, 0), min(d[crit] + self.amp, 1)
        )

        # neighbor.criteria_weights[crit] = max(min(rng.normal(d[crit], self.amp), 1), 0)

        s = sum([w for w in neighbor.criteria_weights.values()])
        for c, w in neighbor.criteria_weights.items():
            neighbor.criteria_weights[c] = w / s

        # weight = int(d[crit] + 0.99)

        # max = max_weight(len(d))

        # if 1 < weight < max:
        #     new_weight = rng.choice([weight - 1, weight + 1])
        # elif 1 == weight:
        #     new_weight = weight + 1
        # elif weight == max:
        #     new_weight = weight - 1
        # else:
        #     raise ValueError(f"{weight} is not between 0 and {max}")

        # d[crit] = new_weight

        return neighbor


class NeighborCapacities(Neighbor[RMP]):
    def __call__(self, model, rng):
        neighbor = deepcopy(model)
        power_set = neighbor.criteria_capacities
        keys = list(power_set)
        crits = keys[rng.choice(len(keys))]
        while power_set.min_capacity(crits) == power_set.max_capacity(crits):
            crits = keys[rng.choice(len(keys))]
        capacity = power_set[crits]
        new_capa = rng.integers(power_set.min_capacity(crits), power_set.max_capacity(crits), endpoint=True)
        # if power_set.min_capacity(crits) < capacity < power_set.max_capacity(crits):
        #     new_capa = rng.choice([capacity - 1, capacity + 1])
        # elif power_set.min_capacity(crits) == capacity:
        #     new_capa = capacity + 1
        # elif capacity == power_set.max_capacity(crits):
        #     new_capa = capacity - 1
        # else:
        #     raise ValueError(
        #         f"{capacity} is not between {power_set.min_capacity(crits)}"
        #         f" and {power_set.max_capacity(crits)}"
        #     )
        neighbor.criteria_capacities[crits] = new_capa
        return neighbor


class NeighborLexOrder(Neighbor[RMP | SRMP]):
    def __call__(self, model, rng):
        neighbor = deepcopy(model)
        lex_order = neighbor.lexicographic_order
        i = rng.choice(len(lex_order))
        j = rng.choice([x for x in range(len(lex_order)) if x != i])
        neighbor.lexicographic_order[i], neighbor.lexicographic_order[j] = (
            lex_order[j],
            lex_order[i],
        )
        # i = rng.choice(len(lex_order) - 1)
        # neighbor.lexicographic_order[i], neighbor.lexicographic_order[i + 1] = (
        #     lex_order[i + 1],
        #     lex_order[i],
        # )
        return neighbor
