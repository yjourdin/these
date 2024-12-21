from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np
from mcda import PerformanceTable
from numpy.random import Generator

from ..dataclass import Dataclass
from ..rmp.model import RMPModel
from ..srmp.model import SRMPModel


class Neighbor[S](ABC):
    @abstractmethod
    def __call__(self, sol: S, rng: Generator) -> S: ...


class RandomNeighbor[S](Neighbor[S]):
    def __init__(
        self,
        neighbors: Sequence[Neighbor[S]],
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


@dataclass
class NeighborProfile(Neighbor[SRMPModel | RMPModel], Dataclass):
    amp: float = 1

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(len(neighbor.profiles.criteria))
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        profile_perf = cast(float, neighbor.profiles.cell[profile_ind, crit_ind])

        profile_perf = rng.uniform(
            max(profile_perf - self.amp, 0), min(profile_perf + self.amp, 1)
        )

        neighbor.profiles.data.iloc[profile_ind, crit_ind] = profile_perf

        neighbor.profiles.data.iloc[:, crit_ind] = neighbor.profiles.data.iloc[
            :, crit_ind
        ].sort_values()

        return neighbor


@dataclass
class NeighborProfileDiscretized(Neighbor[SRMPModel | RMPModel], Dataclass):
    values: PerformanceTable
    local: bool = False

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(len(neighbor.profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(neighbor.profiles.alternatives))
        profile_perf = cast(float, neighbor.profiles.cell[profile_ind, crit_ind])
        profile_perf_ind = cast(int, crit_values[crit_values == profile_perf].index[0])

        if self.local:
            available_ind = []
            if profile_perf_ind > 0:
                available_ind.append(profile_perf_ind - 1)
            if profile_perf_ind < (len(self.values.alternatives) - 1):
                available_ind.append(profile_perf_ind + 1)
        else:
            available_ind = range(len(self.values.alternatives))
        profile_perf_ind = rng.choice(available_ind)

        neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit_values[
            crit_values.index[profile_perf_ind]
        ]

        neighbor.profiles.data.iloc[:, crit_ind] = neighbor.profiles.data.iloc[
            :, crit_ind
        ].sort_values()

        return neighbor


@dataclass
class NeighborWeight(Neighbor[SRMPModel], Dataclass):
    amp: float = 1

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(len(neighbor.weights))
        weight = neighbor.weights[crit_ind]
        weight = rng.uniform(max(weight - self.amp, 0), min(weight + self.amp, 1))
        neighbor.weights[crit_ind] = weight

        neighbor.weights /= sum(neighbor.weights)

        return neighbor


@dataclass
class NeighborWeightDiscretized(Neighbor[SRMPModel]):
    max: int
    local: bool = False

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        crit_ind = rng.choice(len(neighbor.weights))
        weight = neighbor.weights[crit_ind]

        if self.local:
            available_ind = []
            if weight > 0:
                available_ind.append(weight - 1)
            if weight < self.max:
                available_ind.append(weight + 1)
        else:
            available_ind = range(self.max + 1)
        weight = rng.choice(available_ind)

        neighbor.weights[crit_ind] = weight

        return neighbor


@dataclass
class NeighborImportanceRelation(Neighbor[RMPModel]):
    local: bool = False

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        importance_relation = neighbor.importance_relation
        keys = list(importance_relation)
        min_score = max_score = 0
        
        while min_score >= max_score:
            coalition = keys[rng.choice(len(importance_relation))]
            min_score = importance_relation.min(coalition)
            max_score = importance_relation.max(coalition)
        
        if self.local:
            score = importance_relation[coalition]
            available_score = []
            if score > min_score:
                available_score.append(score - 1)
            if score < max_score:
                available_score.append(score + 1)
        else:
            available_score = range(min_score, max_score + 1)
        
        score = rng.choice(available_score)
        neighbor.importance_relation[coalition] = score

        return neighbor


@dataclass
class NeighborLexOrder(Neighbor[SRMPModel | RMPModel], Dataclass):
    local: bool = False

    def __call__(self, sol, rng):
        neighbor = deepcopy(sol)

        lex_order = neighbor.lexicographic_order

        if self.local:
            i = rng.choice(len(lex_order) - 1)
            j = i + 1
        else:
            i = rng.choice(len(lex_order))
            j = rng.choice([x for x in range(len(lex_order)) if x != i])
        neighbor.lexicographic_order[i], neighbor.lexicographic_order[j] = (
            lex_order[j],
            lex_order[i],
        )

        return neighbor
