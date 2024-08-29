from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from itertools import chain, combinations
from typing import Any, Collection, Generic, cast

import numpy as np
from mcda import PerformanceTable
from numpy.random import Generator

from ..rmp.model import RMPModelCapacity
from ..srmp.model import SRMPModel
from .type import Solution


class Neighbor(Generic[Solution], ABC):
    @abstractmethod
    def __call__(self, sol: Solution, rng: Generator) -> Solution: ...


class RandomNeighbor(Neighbor[Solution]):
    def __init__(
        self,
        neighbors: Sequence[Neighbor[Solution]],
        prob: Sequence[float] | None = None,
    ):
        self.neighbors = neighbors
        if prob:
            prob_array = np.array(prob)
            self.prob = prob_array / prob_array.sum()
        else:
            self.prob = None

    def __call__(self, sol, rng):
        i = rng.choice(len(self.neighbors), p=self.prob)
        return self.neighbors[i](sol, rng)


class NeighborProfiles(Neighbor[SRMPModel | RMPModelCapacity]):
    def __init__(self, values: PerformanceTable):
        self.values = values

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        # crit_ind = rng.choice(len(neighbor.profiles.criteria))
        # profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        # value = neighbor.profiles.data.iloc[profile_ind, crit_ind]

        # neighbor.profiles.data.iloc[profile_ind, crit_ind] = rng.uniform(
        #     max(value - self.amp, 0), min(value + self.amp, 1)
        # )

        crit_ind = rng.choice(len(neighbor.profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))

        # value = neighbor.profiles.data.iloc[profile_ind, crit_ind]
        # value_ind = cast(
        #     int, crit_values.index.get_loc(crit_values[crit_values == value].index[0])
        # )
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

        neighbor.profiles.data.iloc[:, crit_ind] = neighbor.profiles.data.iloc[
            :, crit_ind
        ].sort_values()
        return neighbor


class NeighborProfilesSRMP(Neighbor[SRMPModel]):
    def __init__(self, values: PerformanceTable):
        self.values = values

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(
            [
                i
                for i, x in enumerate(neighbor.profiles.criteria)
                if neighbor.weights[x] != 0
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

        neighbor.profiles.data.transform(np.sort)

        return neighbor


class NeighborWeights(Neighbor[SRMPModel]):
    def __init__(self, amp: float):
        self.amp = amp

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(len(neighbor.weights))

        neighbor.weights[crit_ind] = rng.uniform(
            max(neighbor.weights[crit_ind] - self.amp, 0),
            min(neighbor.weights[crit_ind] + self.amp, 1),
        )

        # neighbor.weights[crit_ind] = max(
        #     min(rng.normal(neighbor.weights[crit_ind], self.amp), 1), 0
        # )

        s = sum(neighbor.weights)
        neighbor.weights = [w / s for w in neighbor.weights]

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


class NeighborCapacity(Neighbor[RMPModelCapacity]):
    def __init__(self, s: Collection[Any]):
        s = set(s)

        power_set_tmp = chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1)
        )
        power_set = {frozenset(i for i in ss) for ss in power_set_tmp}

        self.supremum = {ss: {ss | {i} for i in (s - ss)} for ss in power_set}
        self.infimum = {ss: {ss - {i} for i in ss} for ss in power_set}

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        capacities = neighbor.capacity
        keys = list(capacities)
        crits = keys[rng.choice(len(keys))]

        infimum_capacities = [capacities[ss] for ss in self.infimum[crits]]
        min_capacity = max(infimum_capacities) if infimum_capacities else 0
        supremum_capacities = [capacities[ss] for ss in self.supremum[crits]]
        max_capacity = min(supremum_capacities) if supremum_capacities else 1

        new_capa = rng.uniform(min_capacity, max_capacity)
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
        neighbor.capacity[crits] = new_capa
        return neighbor


class NeighborLexOrder(Neighbor[SRMPModel | RMPModelCapacity]):
    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)
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
