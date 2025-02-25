from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from numpy.random import Generator

from ..dataclass import Dataclass
from ..performance_table.type import PerformanceTableType
from ..rmp.importance_relation import ImportanceRelation
from ..rmp.model import RMPModel
from ..rmp.permutation import swap
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

    def __call__(self, sol: S, rng: Generator):
        i = rng.choice(len(self.neighbors), p=self.prob)
        return self.neighbors[i](sol, rng)


@dataclass
class NeighborProfile[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    amp: float = 1

    def __call__(self, sol: S, rng: Generator):
        profiles = deepcopy(sol.profiles)

        crit_ind = rng.choice(len(profiles.criteria))
        profile_ind = rng.choice(len(profiles.alternatives))
        profile_perf = cast(float, profiles.cell[profile_ind, crit_ind])

        profile_perf = rng.uniform(
            max(profile_perf - self.amp, 0), min(profile_perf + self.amp, 1)
        )

        profiles.data.iloc[profile_ind, crit_ind] = profile_perf

        profiles.data.iloc[:, crit_ind] = profiles.data.iloc[:, crit_ind].sort_values()

        return replace(sol, profiles=profiles)


@dataclass
class NeighborProfileDiscretized[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    values: PerformanceTableType
    local: bool = False

    def __call__(self, sol: S, rng: Generator):
        profiles = deepcopy(sol.profiles)

        crit_ind = rng.choice(len(profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]

        profiles_values = sol.profiles.data.iloc[:, crit_ind].to_list()

        profile_ind = rng.choice(len(profiles.alternatives))
        profile_perf = cast(float, profiles.cell[profile_ind, crit_ind])
        profile_perf_ind = cast(int, crit_values[crit_values == profile_perf].index[0])

        if self.local:
            available_ind = []
            if profile_perf_ind > 0:
                available_ind.append(profile_perf_ind - 1)
            if profile_perf_ind < (len(self.values.alternatives) - 1):  # type: ignore
                available_ind.append(profile_perf_ind + 1)
        else:
            available_ind = list(range(len(self.values.alternatives)))  # type: ignore
        profile_perf_ind = rng.choice(available_ind)

        profiles_values[profile_ind] = crit_values[crit_values.index[profile_perf_ind]]

        # profiles.data.iloc[profile_ind, crit_ind] = crit_values[
        #     crit_values.index[profile_perf_ind]
        # ]

        profiles.data.iloc[:, crit_ind] = sorted(profiles_values)

        # profiles.data.iloc[:, crit_ind] = profiles.data.iloc[:, crit_ind].sort_values()

        return replace(sol, profiles=profiles)


@dataclass
class NeighborWeightAmp[S: SRMPModel](Neighbor[S], Dataclass):
    amp: float = 1

    def __call__(self, sol: S, rng: Generator):
        weights = deepcopy(sol.weights)

        crit_ind = rng.choice(len(weights))
        weight = weights[crit_ind]
        weight = rng.uniform(max(weight - self.amp, 0), min(weight + self.amp, 1))
        weights[crit_ind] = weight

        weights /= sum(weights)

        return replace(sol, weights=weights)


@dataclass
class NeighborWeightDiscretized[S: SRMPModel](Neighbor[S]):
    max: int
    local: bool = False

    def __call__(self, sol: S, rng: Generator):
        weights = deepcopy(sol.weights)

        crit_ind = rng.choice(len(weights))
        weight = weights[crit_ind]

        if self.local:
            available_ind = []
            if weight > 0:
                available_ind.append(weight - 1)
            if weight < self.max:
                available_ind.append(weight + 1)
        else:
            available_ind = list(range(self.max + 1))
        weight = rng.choice(available_ind)

        weights[crit_ind] = weight

        return replace(sol, weights=weights)

def weights_local_change(
    weights: npt.NDArray[np.float64],
    crit_ind: int,
    increase: bool = True,
):
    subset_sum = compute_subset_sum(np.delete(weights, crit_ind))

    weight = weights[crit_ind]

    alpha, eq1 = compute_alpha(subset_sum, weight, increase)

    if eq1:
        alpha = (1 + alpha) / 2

    delta = (1 - weight) * (1 - alpha)

    new = alpha * weights
    new[crit_ind] = weight + delta
    return new



def compute_subset_sum(weights: npt.NDArray[np.float64]):
    if len(weights) == 1:
        return weights
    weight = weights[-1]
    subset_sums = compute_subset_sum(weights[:-1])
    return np.concat((subset_sums, np.array([weight]), subset_sums + weight))


@njit(fastmath=True)  # type: ignore
def compute_alpha_increase(subset_sum: npt.NDArray[np.float64], weight: float):
    N = len(subset_sum)
    eq1 = False
    best_denom = np.inf
    for i in range(N):
        w1 = subset_sum[i]
        if w1 > 1e-10:
            denom1 = 2 * w1
            if denom1 < best_denom:
                for j in range(N):
                    if ((i + 1) & (j + 1)) == 0:
                        denom2 = denom1 + subset_sum[j]
                        eq1 |= denom2 == 1
                        if 1 < denom2 < best_denom:
                            best_denom = denom2
    return 1 / best_denom, eq1


@njit(fastmath=True)  # type: ignore
def compute_alpha_decrease(subset_sum: npt.NDArray[np.float64], weight: float):
    N = len(subset_sum)
    eq1 = False
    best_denom = 1 - weight
    for i in range(N):
        w1 = subset_sum[i]
        if w1 > 1e-10:
            denom1 = 2 * w1
            if denom1 < 1:
                for j in range(N):
                    if ((i + 1) & (j + 1)) == 0:
                        denom2 = denom1 + subset_sum[j]
                        eq1 |= denom2 == 1
                        if best_denom < denom2 < 1:
                            best_denom = denom2
    return 1 / best_denom, eq1


def compute_alpha(subset_sum: npt.NDArray[np.float64], weight: float, increase: bool):
    f = compute_alpha_increase if increase else compute_alpha_decrease
    return f(subset_sum, weight)


@dataclass
class NeighborWeight[S: SRMPModel](Neighbor[S]):
    def __call__(self, sol: S, rng: Generator):
        crit_ind = rng.choice(len(sol.weights))

        subset_sum = compute_subset_sum(np.delete(sol.weights, crit_ind))

        if (weight := sol.weights[crit_ind]) == 0:
            increase = True
        elif weight == 1:
            increase = False
        else:
            increase = bool(rng.choice(2))

        alpha, eq1 = compute_alpha(subset_sum, weight, increase)

        if eq1:
            alpha = (1 + alpha) / 2

        delta = (1 - weight) * (1 - alpha)

        new_weights = alpha * sol.weights
        new_weights[crit_ind] = weight + delta

        return replace(sol, weights=new_weights)


@dataclass
class NeighborImportanceRelation[S: RMPModel](Neighbor[S]):
    local: bool = False

    def __call__(self, sol: S, rng: Generator):
        importance_relation: ImportanceRelation = deepcopy(sol.importance_relation)

        keys = list(importance_relation)
        min_score = max_score = 0

        coalition = cast(frozenset[int], None)
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
            available_score = list(range(min_score, max_score + 1))

        score = rng.choice(available_score)
        importance_relation[coalition] = score

        return replace(sol, importance_relation=importance_relation)


@dataclass
class NeighborLexOrder[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    local: bool = False

    def __call__(self, sol: S, rng: Generator):
        lexicographic_order = deepcopy(sol.lexicographic_order)

        if self.local:
            i = rng.choice(len(lexicographic_order) - 1)
            j = i + 1
        else:
            i = rng.choice(len(lexicographic_order))
            j = rng.choice([x for x in range(len(lexicographic_order)) if x != i])
        swap(lexicographic_order, i, j)

        return replace(sol, lexicographic_order=lexicographic_order)
