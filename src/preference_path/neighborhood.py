from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, replace
from itertools import chain, product
from typing import cast

import numpy as np
from mcda import PerformanceTable
from more_itertools import powerset
from numpy.random import Generator

from ..constants import DECIMALS
from ..dataclass import Dataclass
from ..random import rng
from ..rmp.permutation import adjacent_swap
from ..sa.neighbor import weights_local_change
from ..srmp.model import FrozenSRMPModel
from ..utils import midpoints


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
        self.rng.shuffle(neighbors)  # type: ignore
        return neighbors


@dataclass
class NeighborhoodProfile(Neighborhood[FrozenSRMPModel], Dataclass):
    midpoints: PerformanceTable = field(init=False)
    alternatives: InitVar[PerformanceTable]

    def __post_init__(self, alternatives: PerformanceTable):
        self.midpoints = midpoints(alternatives)

    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []

        for profile_ind, profile in enumerate(sol.profiles):
            for crit_ind, crit in self.midpoints.data.items():
                crit_ind = cast(int, crit_ind)
                indices = (
                    np.searchsorted(crit.to_numpy(), profile[crit_ind], "left") - 1,
                    np.searchsorted(crit.to_numpy(), profile[crit_ind], "right"),
                )

                bounds: list[float] = [0, 1]
                if profile_ind > 0:
                    bounds[0] = sol.profiles[profile_ind - 1][crit_ind]
                if profile_ind < len(sol.profiles) - 1:
                    bounds[1] = sol.profiles[profile_ind + 1][crit_ind]

                for ind in indices:
                    if (0 <= ind < len(crit)) and (
                        bounds[0] <= (new_value := crit.iloc[ind]) <= bounds[1]
                    ):
                        result.append(
                            replace(
                                sol,
                                profiles=tuple(
                                    sol.profiles[i]
                                    if i != profile_ind
                                    else tuple(
                                        profile[j]
                                        if j != crit_ind
                                        else float(new_value)
                                        for j in range(len(profile))
                                    )
                                    for i in range(len(sol.profiles))
                                ),
                            )
                        )

        return result


@dataclass
class NeighborhoodWeight(Neighborhood[FrozenSRMPModel], Dataclass):
    powersets: tuple[tuple[int, ...], ...] = field(init=False)
    nb_crit: InitVar[int]

    def __post_init__(self, nb_crit: int):
        self.powersets = tuple(powerset(range(nb_crit)))[1:-1]

    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []

        for crit, increase in product(range(len(sol.weights)), [False, True]):
            if (
                weights := weights_local_change(
                    self.powersets, np.array(sol.weights), crit, increase
                )
            ) is not None:
                if np.array_equal(weights := weights.round(DECIMALS), sol.weights):
                    result.append(replace(sol, weights=weights))

        return result


class NeighborhoodLexOrder(Neighborhood[FrozenSRMPModel]):
    def __call__(self, sol):
        result: list[FrozenSRMPModel] = []

        for i in range(len(sol.lexicographic_order) - 1):
            result.append(
                replace(
                    sol,
                    lexicographic_order=tuple(
                        adjacent_swap(list(sol.lexicographic_order), i)
                    ),
                )
            )

        return result
