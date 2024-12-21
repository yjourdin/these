from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from itertools import chain

import numpy as np
from mcda import PerformanceTable
from more_itertools import powerset_of_sets

from src.utils import midpoints

from ..dataclass import Dataclass
from ..srmp.model import SRMPModel


class Neighborhood[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> list[S]: ...


@dataclass
class NeighborhoodCombined[S](Neighborhood[S], Dataclass):
    neighborhoods: list[Neighborhood[S]] = field(default_factory=list)

    def __call__(self, sol: S) -> list[S]:
        return list(
            chain.from_iterable(
                neighborhood(sol) for neighborhood in self.neighborhoods
            )
        )


@dataclass
class NeighborhoodProfile(Neighborhood[SRMPModel], Dataclass):
    midpoints: PerformanceTable = field(init=False)
    alternatives: InitVar[PerformanceTable]

    def __post_init__(self, alternatives: PerformanceTable):
        self.midpoints = midpoints(alternatives)

    def __call__(self, sol: SRMPModel) -> list[SRMPModel]:
        result = []

        for profile_ind, profile in sol.profiles.alternatives_values.items():
            for crit_ind, crit in self.midpoints.criteria_values.items():
                lower_ind = None
                upper_ind = None
                for profile_perf_ind, crit_value in crit.sort(True).items():
                    if crit_value.value >= profile[crit_ind]:
                        upper_ind = profile_perf_ind
                        if crit_value.value == profile[crit_ind]:
                            upper_ind += 1
                        break
                    else:
                        lower_ind = profile_perf_ind
                        upper_ind = profile_perf_ind + 1

                if crit_value.value == profile[crit_ind]:
                    if lower_ind is not None:
                        neighbor = deepcopy(sol)
                        neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit[
                            lower_ind
                        ].value
                        result.append(neighbor)

                    if upper_ind is not None:
                        neighbor = deepcopy(sol)
                        neighbor.profiles.data.iloc[profile_ind, crit_ind] = crit[
                            upper_ind
                        ].value
                        result.append(neighbor)

        return result


@dataclass
class NeighborhoodWeight(Neighborhood[SRMPModel], Dataclass):
    powersets: list[set[int]] = field(init=False)
    nb_crit: InitVar[int]

    def __post_init__(self, nb_crit: int):
        self.powersets = list(powerset_of_sets(range(nb_crit)))

    def __call__(self, sol: SRMPModel) -> list[SRMPModel]:
        result = []

        for crit, weight in enumerate(sol.weights):
            if weight == 1:
                neighbor = deepcopy(sol)
                neighbor.weights = np.full_like(sol.weights, 0.5)
                result.append(neighbor)
            else:
                with_crit = np.array([])
                without_crit = np.array([])

                for set in self.powersets:
                    if crit in set:
                        np.append(with_crit, sol.weights[list(set)].sum())
                    else:
                        np.append(without_crit, sol.weights[list(set)].sum())

                diff = np.subtract.outer(without_crit, with_crit)
                progress_factor = np.add.outer(
                    without_crit / (1 - weight), 1 - (with_crit - weight) / (1 - weight)
                )

                progress = diff / progress_factor

                increase = progress[progress > 0].min(initial=np.inf)
                decrease = progress[progress < 0].max(initial=-np.inf)

                if 0 in progress:
                    increase /= 2
                    decrease /= 2

                if increase < 1 - weight:
                    neighbor = deepcopy(sol)
                    neighbor.weights -= (
                        sol.weights * (1 - weight - increase) / (1 - weight)
                    )
                    neighbor.weights[crit] = sol.weights[crit] + increase
                    result.append(neighbor)
                if decrease > -weight:
                    neighbor = deepcopy(sol)
                    neighbor.weights -= (
                        sol.weights * (1 - weight - decrease) / (1 - weight)
                    )
                    neighbor.weights[crit] = sol.weights[crit] + decrease
                    result.append(neighbor)

        return result


class NeighborhoodLexOrder(Neighborhood[SRMPModel]):
    def __call__(self, sol: SRMPModel) -> list[SRMPModel]:
        result = []

        for i in range(len(sol.lexicographic_order) - 1):
            neighbor = deepcopy(sol)
            neighbor.lexicographic_order[i], neighbor.lexicographic_order[i + 1] = (
                neighbor.lexicographic_order[i + 1],
                neighbor.lexicographic_order[i],
            )
            result.append(neighbor)

        return result
