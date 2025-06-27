from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, replace
from itertools import chain, product
from typing import cast

import numpy as np
import numpy.typing as npt
from mcda.relations import PreferenceStructure
from pandas import Series

from ..dataclass import Dataclass
from ..performance_table.type import PerformanceTableType
from ..preference_structure.fitness import comparisons_ranking
from ..random import RNGParam, rng_
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
    rng: InitVar[RNGParam] = None

    def __post_init__(self, rng: RNGParam = None):
        self._rng = rng_(rng)

    def __call__(self, sol: S):
        neighbors = list(
            chain.from_iterable(
                neighborhood(sol) for neighborhood in self.neighborhoods
            )
        )
        self._rng.shuffle(neighbors)  # type: ignore
        return neighbors


@dataclass
class NeighborhoodProfile(Neighborhood[FrozenSRMPModel], Dataclass):
    midpoints: PerformanceTableType = field(init=False)
    alternatives: PerformanceTableType
    target_preferences: PreferenceStructure

    def __post_init__(self):
        self.midpoints = midpoints(self.alternatives)

    def __call__(self, sol: FrozenSRMPModel):
        result: list[FrozenSRMPModel] = []

        relevant_values = np.sort(
            cast(
                npt.NDArray[np.float64],
                self.alternatives.subtable(
                    PreferenceStructure(
                        comparisons_ranking(
                            self.target_preferences,
                            sol.model.rank_series(self.alternatives).to_dict(),
                        )
                    ).elements
                ).data.to_numpy(),
            ),
            0,
        )
        # if any(
        #     np.any(
        #         np.equal.outer(
        #             self.midpoints.data.to_numpy()[:, i], relevant_values[:, i]
        #         )
        #     )
        #     for i in range(3)
        # ):
        #     print((self.midpoints.data.to_numpy(), relevant_values))

        for profile_ind, profile in enumerate(sol.profiles):
            profile = cast(tuple[float, ...], profile)
            for crit_ind, crit in self.midpoints.data.items():
                crit = cast("Series[float]", crit)
                crit_ind = cast(int, crit_ind)
                crit_numpy: npt.NDArray[np.float64] = crit.to_numpy()

                relevant_bounds = (
                    np.max(
                        relevant_values[:, crit_ind][
                            relevant_values[:, crit_ind] < profile[crit_ind]
                        ],
                        initial=0,
                    ),
                    np.min(
                        relevant_values[:, crit_ind][
                            relevant_values[:, crit_ind] > profile[crit_ind]
                        ],
                        initial=1,
                    ),
                )

                new_values = (
                    np.max(crit_numpy[crit_numpy <= relevant_bounds[0]], initial=-1),
                    np.min(crit_numpy[crit_numpy >= relevant_bounds[1]], initial=2),
                )

                profile_bounds = (
                    sol.profiles[profile_ind - 1][crit_ind] if profile_ind > 0 else 0,
                    sol.profiles[profile_ind + 1][crit_ind]
                    if profile_ind < len(sol.profiles) - 1
                    else 1,
                )

                # if profile_ind == 0 and crit_ind == 0:
                #     print(profile[crit_ind], relevant_bounds, new_values)

                for new_value in new_values:
                    if profile_bounds[0] <= new_value <= profile_bounds[1]:
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

        # print(
        #     PreferenceStructure(
        #         comparisons_ranking(
        #             self.target_preferences,
        #             sol.model.rank_series(self.alternatives).to_dict(),
        #         )
        #     ),
        #     sol.profiles[-1][-1],
        #     relevant_bounds,
        #     new_values,
        # )
        return result


@dataclass
class NeighborhoodWeight(Neighborhood[FrozenSRMPModel]):
    def __call__(self, sol: FrozenSRMPModel):
        result: list[FrozenSRMPModel] = []

        for crit, increase in product(range(len(sol.weights)), [False, True]):
            result.append(
                replace(
                    sol,
                    weights=weights_local_change(np.array(sol.weights), crit, increase),
                )
            )

        return result


class NeighborhoodLexOrder(Neighborhood[FrozenSRMPModel]):
    def __call__(self, sol: FrozenSRMPModel):
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
