from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from itertools import chain
from typing import cast

import numpy as np
from mcda import PerformanceTable
from more_itertools import powerset_of_sets
from numpy.random import Generator

from ..dataclass import Dataclass
from ..rmp.permutation import adjacent_swap
from ..srmp.model import FrozenSRMPModel
from ..utils import midpoints
from ..random import rng

class Neighborhood[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> list[S]: ...


@dataclass
class NeighborhoodCombined[S](Neighborhood[S], Dataclass):
    neighborhoods: list[Neighborhood[S]] = field(default_factory=list)
    rng: Generator = rng()

    def __call__(self, sol: S):
        neighbors = list(
            chain.from_iterable(
                neighborhood(sol) for neighborhood in self.neighborhoods
            )
        )
        self.rng.shuffle(neighbors) # type: ignore
        return neighbors


@dataclass
class NeighborhoodProfile(Neighborhood[FrozenSRMPModel], Dataclass):
    midpoints: PerformanceTable = field(init=False)
    alternatives: InitVar[PerformanceTable]

    def __post_init__(self, alternatives: PerformanceTable):
        self.midpoints = midpoints(alternatives)

    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []
        sol_mutable = sol.model

        for profile_ind, profile in sol_mutable.profiles.alternatives_values.items():
            for crit_ind, crit in self.midpoints.data.items():
                crit_ind = cast(int, crit_ind)
                indices = (
                    np.searchsorted(crit.to_numpy(), profile[crit_ind].value, "left")
                    - 1,
                    np.searchsorted(crit.to_numpy(), profile[crit_ind].value, "right"),
                )

                bounds: list[float] = [0, 1]
                if profile_ind > 0:
                    bounds[0] = cast(
                        float, sol_mutable.profiles.data.iloc[profile_ind - 1, crit_ind]
                    )
                if profile_ind < len(sol.profiles) - 1:
                    bounds[1] = cast(
                        float, sol_mutable.profiles.data.iloc[profile_ind + 1, crit_ind]
                    )

                for ind in indices:
                    if (0 <= ind < len(crit)) and (
                        bounds[0] <= (new_value := crit.iloc[ind]) <= bounds[1]
                    ):
                        neighbor = deepcopy(sol_mutable)
                        neighbor.profiles.data.iloc[profile_ind, crit_ind] = new_value
                        result.append(neighbor.frozen)

        return result


@dataclass
class NeighborhoodWeight(Neighborhood[FrozenSRMPModel], Dataclass):
    powersets: list[set[int]] = field(init=False)
    nb_crit: InitVar[int]

    def __post_init__(self, nb_crit: int):
        self.powersets = list(powerset_of_sets(range(nb_crit)))[1:-1]

    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []
        sol_mutable = sol.model

        for crit, weight in enumerate(sol_mutable.weights):
            if weight == 1:
                neighbor = deepcopy(sol_mutable)
                neighbor.weights = np.full_like(sol_mutable.weights, 0.5)
                result.append(neighbor.frozen)
            else:
                with_crit = []
                without_crit = []
                for set in self.powersets:
                    if weights_sum := sol_mutable.weights[list(set)].sum():
                        if crit in set:
                            with_crit.append(weights_sum)
                        else:
                            without_crit.append(weights_sum)

                with_crit_np = np.array(with_crit)
                without_crit_np = np.array(without_crit)

                diff = np.subtract.outer(without_crit_np, with_crit_np)
                progress_factor = np.add.outer(
                    without_crit_np / (1 - weight),
                    1 - (with_crit_np - weight) / (1 - weight),
                )

                progress = diff / progress_factor

                increase = progress[progress > 0].min(initial=np.inf)
                decrease = progress[progress < 0].max(initial=-np.inf)

                if 0 in progress:
                    increase /= 2
                    decrease /= 2

                if increase < 1 - weight:
                    neighbor = deepcopy(sol_mutable)
                    neighbor.weights -= increase * (sol_mutable.weights / (1 - weight))
                    neighbor.weights[crit] = sol_mutable.weights[crit] + increase
                    result.append(neighbor.frozen)
                if decrease > -weight:
                    neighbor = deepcopy(sol_mutable)
                    neighbor.weights -= decrease * (sol_mutable.weights / (1 - weight))
                    neighbor.weights[crit] = sol_mutable.weights[crit] + decrease
                    result.append(neighbor.frozen)

        return result


class NeighborhoodLexOrder(Neighborhood[FrozenSRMPModel]):
    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []

        for i in range(len(sol.lexicographic_order) - 1):
            neighbor = deepcopy(sol.model)
            adjacent_swap(neighbor.lexicographic_order, i)
            result.append(neighbor.frozen)

        return result
