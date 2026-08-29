from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from functools import reduce
from typing import Literal, cast

import numpy as np
from numba import njit

from src.constants import EPSILON
from src.dataclass import Dataclass, dataclass, replace
from src.performance_table.type import PerformanceTableType
from src.random import RNGParam, rng_, seed_
from src.rmp.importance_relation import ImportanceRelation
from src.rmp.model import RMPModel
from src.rmp.permutation import swap
from src.srmp.model import SRMPModel

from ..utils import kendalltau_distance


class Neighbor[S](ABC):
    @abstractmethod
    def __call__(self, sol: S, rng: RNGParam = None) -> S: ...


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

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
        i = rng.choice(len(self.neighbors), p=self.prob)
        return self.neighbors[i](sol, rng)


@dataclass
class NeighborAccept[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    neighbor: Neighbor[S]
    reference: S
    profile_amp: float
    importance_relation_amp: float
    lexicographic_amp: float

    def profile_accept(self, sol: S):
        return np.all(
            abs(
                sol.profiles.data.to_numpy()
                - self.reference.profiles.data.to_numpy()
            )
            <= self.profile_amp
        )

    def importance_relation_accept(self, sol: S):
        return sum(
            abs(v - self.reference.importance_relation.get(k, v))
            for k, v in sol.importance_relation.items()
        ) <= self.importance_relation_amp * len(sol.importance_relation)

    def lexicographic_order_accept(self, sol: S):
        return (
            kendalltau_distance(
                sol.lexicographic_order, self.reference.lexicographic_order
            )
            <= self.lexicographic_amp
        )

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
        new = self.neighbor(sol, seed_(rng))
        while (
            (not self.profile_accept(new))
            or (not self.importance_relation_accept(new))
            or (not self.lexicographic_order_accept(new))
        ):
            new = self.neighbor(sol, seed_(rng))
        return new


@dataclass
class NeighborProfile[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    amp: float = 1

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
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

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
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
            if profile_perf_ind < (len(self.values.alternatives) - 1):  # pyright: ignore[reportUnknownArgumentType]
                available_ind.append(profile_perf_ind + 1)
        else:
            available_ind = list(range(len(self.values.alternatives)))  # pyright: ignore[reportUnknownArgumentType]
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

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
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

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
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
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    crit_ind: int,
    increase: bool = True,
):
    subset_sum = compute_subset_sum(np.delete(weights, crit_ind))

    weight: float = weights[crit_ind]

    if weight >= 1 - EPSILON:
        return np.full_like(weights, 1 / len(weights))

    alpha, eq1 = compute_alpha(subset_sum, weight, increase)

    if eq1:
        alpha = (1 + alpha) / 2

    delta = (1 - weight) * (1 - alpha)

    new = alpha * weights
    new[crit_ind] = weight + delta

    if new[crit_ind] < EPSILON:
        new[crit_ind] = 0
        new /= np.sum(new)

    mask = (float(new[crit_ind]) - EPSILON < new) & (
        new < float(new[crit_ind]) + EPSILON
    )
    new[mask] = np.sum(new[mask]) / np.sum(mask)

    return new


def add_subset_sum(
    subset_sums: np.ndarray[tuple[int], np.dtype[np.float64]], weight: float
):
    return np.concat((subset_sums, np.array([weight]), subset_sums + weight))


def compute_subset_sum(weights: np.ndarray[tuple[int], np.dtype[np.float64]]):
    return np.concat((np.zeros(1), reduce(add_subset_sum, weights, np.empty(0))))


@njit
def compute_alpha_increase(
    subset_sum: np.ndarray[tuple[int], np.dtype[np.float64]], weight: float
):
    N = len(subset_sum)
    eq1 = False
    best_denom = np.inf
    for i in range(N):
        w1: float = subset_sum[i]
        if w1 > EPSILON:
            denom1 = 2 * w1
            if denom1 < best_denom:
                for j in range(N):
                    if (i & j) == 0:
                        denom2: float = denom1 + subset_sum[j]
                        eq1 |= denom2 == 1
                        if 1 < denom2 < best_denom:
                            best_denom = denom2
    return 1 / best_denom, eq1


@njit
def compute_alpha_decrease(
    subset_sum: np.ndarray[tuple[int], np.dtype[np.float64]], weight: float
):
    N = len(subset_sum)
    eq1 = False
    best_denom = 1 - weight
    for i in range(N):
        w1: float = subset_sum[i]
        if w1 > EPSILON:
            denom1 = 2 * w1
            if denom1 < 1:
                for j in range(N):
                    if (i & j) == 0:
                        denom2: float = denom1 + subset_sum[j]
                        eq1 |= denom2 == 1
                        if best_denom < denom2 < 1:
                            best_denom = denom2
    return 1 / best_denom, eq1


def compute_alpha(
    subset_sum: np.ndarray[tuple[int], np.dtype[np.float64]],
    weight: float,
    increase: bool,
):
    f = compute_alpha_increase if increase else compute_alpha_decrease
    return f(subset_sum, weight)


# @dataclass
# class NeighborWeight[S: SRMPModel](Neighbor[S]):
#     def __call__(self, sol: S, rng: Generator):
#         crit_ind = rng.choice(len(sol.weights))

#         subset_sum = compute_subset_sum(np.delete(sol.weights, crit_ind))

#         if (weight := sol.weights[crit_ind]) == 0:
#             increase = True
#         elif weight == 1:
#             increase = False
#         else:
#             increase = bool(rng.choice(2))

#         alpha, eq1 = compute_alpha(subset_sum, weight, increase)

#         if eq1:
#             alpha = (1 + alpha) / 2

#         delta = (1 - weight) * (1 - alpha)

#         new_weights = alpha * sol.weights
#         new_weights[crit_ind] = weight + delta

#         return replace(sol, weights=new_weights)


@dataclass
class NeighborWeight[S: SRMPModel](Neighbor[S]):
    def __call__(self, sol: S, rng: RNGParam = None):
        if np.any(sol.weights >= 1 - EPSILON):
            new_weights = np.full_like(sol.weights, 1 / len(sol.weights))
        else:
            rng = rng_(rng)

            crit_ind = rng.choice(len(sol.weights))

            weight: float = sol.weights[crit_ind]

            subset_sum = compute_subset_sum(np.delete(sol.weights, crit_ind))

            diffs = np.diff(np.sort(add_subset_sum(subset_sum, weight)))

            eps = np.min(diffs, initial=1, where=diffs != 0)

            d: Literal[-1, 0, 1] = rng.choice([-1, 0, 1])

            s = rng.integers(1, len(subset_sum))

            i = rng.choice(len(subset_sum))

            j1 = s & i

            j2 = s & (~i)

            s_min, s_max = (
                (s1, s2)
                if (s1 := float(subset_sum[j1])) <= (s2 := float(subset_sum[j2]))
                else (s2, s1)
            )

            s_zero = 1 - weight - s_min - s_max

            alpha = (1 + d * eps) / (2 * s_max + s_zero)

            delta = (1 - weight) * (1 - alpha)

            new_weights = alpha * sol.weights
            new_weights[crit_ind] = weight + delta

            if new_weights[crit_ind] < EPSILON:
                new_weights[crit_ind] = 0
                new_weights /= np.sum(new_weights)

            mask = (float(new_weights[crit_ind]) - EPSILON < new_weights) & (
                new_weights < float(new_weights[crit_ind]) + EPSILON
            )
            new_weights[mask] = np.sum(new_weights[mask]) / np.sum(mask)

        return replace(sol, weights=new_weights)


@dataclass
class NeighborImportanceRelation[S: RMPModel](Neighbor[S]):
    local: bool = False

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
        importance_relation: ImportanceRelation = deepcopy(sol.importance_relation)

        keys = list(importance_relation)
        min_score = max_score = 0

        coalition = cast(frozenset[int], None)
        while min_score >= max_score:
            coalition = keys[rng.choice(len(importance_relation))]
            min_score = importance_relation.min(coalition)
            max_score = importance_relation.max(coalition)

        score = importance_relation[coalition]
        if self.local:
            available_scores = np.array([])
            if score > min_score:
                available_scores = np.append(available_scores, score - 1)
            if score < max_score:
                available_scores = np.append(available_scores, score + 1)
        else:
            scores = np.array(list(importance_relation.values()), dtype=np.float64)
            available_scores = np.unique(
                np.clip([scores, scores - 1, scores + 1], min_score, max_score)
            )

        while score == importance_relation[coalition]:
            score = rng.choice(available_scores)
        importance_relation[coalition] = score

        importance_relation.rerank()

        return replace(sol, importance_relation=importance_relation)


@dataclass
class NeighborLexOrder[S: SRMPModel | RMPModel](Neighbor[S], Dataclass):
    local: bool = False

    def __call__(self, sol: S, rng: RNGParam = None):
        rng = rng_(rng)
        lexicographic_order = deepcopy(sol.lexicographic_order)

        if self.local:
            i = rng.choice(len(lexicographic_order) - 1)
            j = i + 1
        else:
            i = rng.choice(len(lexicographic_order))
            j = rng.choice([x for x in range(len(lexicographic_order)) if x != i])
        swap(lexicographic_order, i, j)

        return replace(sol, lexicographic_order=lexicographic_order)
