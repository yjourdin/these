from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from typing import cast

import numpy as np
import numpy.typing as npt
from mcda import PerformanceTable
from more_itertools import powerset
from numpy.random import Generator

from ..dataclass import Dataclass
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

    def __call__(self, sol, rng):
        i = rng.choice(len(self.neighbors), p=self.prob)
        return self.neighbors[i](sol, rng)


@dataclass
class NeighborProfile(Neighbor[SRMPModel | RMPModel], Dataclass):
    amp: float = 1

    def __call__(self, sol, rng):
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
class NeighborProfileDiscretized(Neighbor[SRMPModel | RMPModel], Dataclass):
    values: PerformanceTable
    local: bool = False

    def __call__(self, sol, rng):
        profiles = deepcopy(sol.profiles)

        crit_ind = rng.choice(len(profiles.criteria))
        crit_values = self.values.data.iloc[:, crit_ind]
        profile_ind = rng.choice(len(profiles.alternatives))
        profile_perf = cast(float, profiles.cell[profile_ind, crit_ind])
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

        profiles.data.iloc[profile_ind, crit_ind] = crit_values[
            crit_values.index[profile_perf_ind]
        ]

        profiles.data.iloc[:, crit_ind] = profiles.data.iloc[:, crit_ind].sort_values()

        return replace(sol, profiles=profiles)


@dataclass
class NeighborWeightAmp(Neighbor[SRMPModel], Dataclass):
    amp: float = 1

    def __call__(self, sol, rng):
        weights = deepcopy(sol.weights)

        crit_ind = rng.choice(len(weights))
        weight = weights[crit_ind]
        weight = rng.uniform(max(weight - self.amp, 0), min(weight + self.amp, 1))
        weights[crit_ind] = weight

        weights /= sum(weights)

        return replace(sol, weights=weights)


@dataclass
class NeighborWeightDiscretized(Neighbor[SRMPModel]):
    max: int
    local: bool = False

    def __call__(self, sol, rng):
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
            available_ind = range(self.max + 1)
        weight = rng.choice(available_ind)

        weights[crit_ind] = weight

        return replace(sol, weights=weights)


def weights_local_change(
    powersets: tuple[tuple[int, ...], ...],
    weights: npt.NDArray[np.float64],
    crit_ind: int,
    increase: bool = True,
):
    weight: float = weights[crit_ind]

    if weight == 1:
        if not increase:
            return np.full_like(weights, 1 / len(weights), float)
    else:
        with_crit = []
        without_crit = []
        for set in powersets:
            weights_sum = weights[list(set)].sum()
            if (len(set) == 0) or (len(set) == len(weights)) or (crit_ind not in set):
                without_crit.append(weights_sum)
            else:
                with_crit.append(weights_sum)
        with_crit_np = np.array(with_crit)
        without_crit_np = np.array(without_crit)

        diff = np.subtract.outer(without_crit_np, with_crit_np)

        progress_factor = np.add.outer(
            np.pad(without_crit_np[1:-1] / (1 - weight), 1),
            1 - (with_crit_np - weight) / (1 - weight),
        )

        progress = diff[progress_factor != 0] / progress_factor[progress_factor != 0]

        change: npt.NDArray[np.float64]
        if increase:
            change = progress[progress > 0].min(initial=np.inf)
        else:
            change = progress[progress < 0].max(initial=-np.inf)

        if 0 in progress:
            change /= 2

        if -weight < change < 1 - weight:
            new = weights - change * (weights / (1 - weight))
            new[crit_ind] = weights[crit_ind] + change
            return new


@dataclass
class NeighborWeight(Neighbor[SRMPModel], Dataclass):
    powersets: tuple[tuple[int, ...], ...] = field(init=False)
    nb_crit: InitVar[int]

    def __post_init__(self, nb_crit: int):
        self.powersets = tuple(powerset(range(nb_crit)))[1:-1]

    def __call__(self, sol, rng):
        weights = deepcopy(sol.weights)

        while (
            new_weights := weights_local_change(
                self.powersets, weights, rng.choice(len(weights)), bool(rng.choice(2))
            )
        ) is None:
            pass

        return replace(sol, weights=new_weights)


@dataclass
class NeighborImportanceRelation(Neighbor[RMPModel]):
    local: bool = False

    def __call__(self, sol, rng):
        importance_relation = deepcopy(sol.importance_relation)

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
        importance_relation[coalition] = score

        return replace(sol, importance_relation=importance_relation)


@dataclass
class NeighborLexOrder(Neighbor[SRMPModel | RMPModel], Dataclass):
    local: bool = False

    def __call__(self, sol, rng):
        lexicographic_order = deepcopy(sol.lexicographic_order)

        if self.local:
            i = rng.choice(len(lexicographic_order) - 1)
            j = i + 1
        else:
            i = rng.choice(len(lexicographic_order))
            j = rng.choice([x for x in range(len(lexicographic_order)) if x != i])
        swap(lexicographic_order, i, j)

        return replace(sol, lexicographic_order=lexicographic_order)
