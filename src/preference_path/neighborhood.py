from abc import ABC, abstractmethod
from dataclasses import InitVar
from itertools import chain, product
from typing import cast

import numpy as np
from mcda.relations import PreferenceStructure
from pandas import Series
from scipy.stats import rankdata

from src.dataclass import Dataclass, dataclass, field, replace
from src.performance_table.type import PerformanceTableType
from src.preference_structure.fitness import comparisons_ranking
from src.random import RNGParam, rng_
from src.rmp.permutation import adjacent_swap
from src.sa.neighbor import weights_local_change
from src.srmp.model import FrozenSRMPModel
from src.utils import midpoints

from ..rmp.model import FrozenRMPModel


class Neighborhood[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> list[S]: ...


@dataclass
class NeighborhoodCombined[S](Neighborhood[S], Dataclass):
    neighborhoods: list[Neighborhood[S]] = field(default_factory=list)
    rng: InitVar[RNGParam] = None

    def __post_init__(self, rng: RNGParam):
        self._rng = rng_(rng)

    def __call__(self, sol: S):
        neighbors = list(
            chain.from_iterable(
                neighborhood(sol) for neighborhood in self.neighborhoods
            )
        )
        # print([len(neighborhood(sol)) for neighborhood in self.neighborhoods])
        self._rng.shuffle(neighbors)
        return neighbors


@dataclass
class NeighborhoodModel[T: FrozenRMPModel | FrozenSRMPModel](
    Neighborhood[T], Dataclass
):
    alternatives: PerformanceTableType | None = None
    target_preferences: PreferenceStructure | None = None

    def different_preferences(self, sol: T):
        return (
            PreferenceStructure(
                comparisons_ranking(
                    self.target_preferences,
                    sol.model.rank_series(self.alternatives).to_dict(),
                )
            )
            if self.alternatives and self.target_preferences
            else None
        )


@dataclass
class NeighborhoodProfile[T: FrozenRMPModel | FrozenSRMPModel](
    NeighborhoodModel[T], Dataclass
):
    midpoints: PerformanceTableType = field(init=False)

    def __post_init__(self):
        assert self.alternatives
        self.midpoints = midpoints(self.alternatives)

    def __call__(self, sol: T):
        result: list[T] = []

        assert self.alternatives

        differences = self.different_preferences(sol)

        relevant_alternatives = np.sort(
            cast(
                np.ndarray[tuple[int, int], np.dtype[np.float64]],
                (
                    self.alternatives.subtable(differences.elements)
                    if differences is not None
                    else self.alternatives
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

        if differences is not None:
            profile_ind_max = 0
            for rel in differences:
                a = np.array([
                    v.value
                    for v in self.alternatives.alternatives_values[rel.a].values()
                ])
                b = np.array([
                    v.value
                    for v in self.alternatives.alternatives_values[rel.b].values()
                ])

                profile_ind = 0
                profile = np.array(sol.profiles[sol.lexicographic_order[profile_ind]])

                while (np.all((a >= profile) == (b >= profile))) and (
                    profile_ind < len(sol.profiles) - 1
                ):
                    profile_ind += 1
                    profile = sol.profiles[sol.lexicographic_order[profile_ind]]

                profile_ind_max = max(profile_ind_max, profile_ind)
        else:
            profile_ind_max = len(sol.profiles) - 1

        for profile_ind in sol.lexicographic_order[: profile_ind_max + 1]:
            profile = sol.profiles[profile_ind]
            for crit_ind, crit in self.midpoints.data.items():
                crit = cast("Series[float]", crit)
                crit_ind = cast(int, crit_ind)
                crit_numpy = crit.to_numpy()

                relevant_bounds = (
                    np.max(
                        relevant_alternatives[:, crit_ind][
                            relevant_alternatives[:, crit_ind] < profile[crit_ind]
                        ],
                        initial=-1,
                    ),
                    np.min(
                        relevant_alternatives[:, crit_ind][
                            relevant_alternatives[:, crit_ind] > profile[crit_ind]
                        ],
                        initial=2,
                    ),
                )

                new_values: list[np.floating] = []
                if np.any(relevant_mask := (crit_numpy < relevant_bounds[0])):
                    new_values.append(np.max(crit_numpy[relevant_mask]))
                if np.any(relevant_mask := (crit_numpy > relevant_bounds[1])):
                    new_values.append(np.min(crit_numpy[relevant_mask]))

                profile_bounds = (
                    sol.profiles[profile_ind - 1][crit_ind] if profile_ind > 0 else 0,
                    sol.profiles[profile_ind + 1][crit_ind]
                    if profile_ind < len(sol.profiles) - 1
                    else 1,
                )

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

        return result


@dataclass
class NeighborhoodImportanceRelation(NeighborhoodModel[FrozenRMPModel]):
    def bounds(
        self,
        importance_relation: tuple[tuple[frozenset[int], float], ...],
        coalition: frozenset[int],
    ):
        try:
            m = max(v for k, v in importance_relation if k < coalition)
        except ValueError:
            m = 1

        try:
            M = min(v for k, v in importance_relation if coalition < k)
        except ValueError:
            M = min(
                max(v for _, v in importance_relation) + 1,
                len(importance_relation),
            )

        # for i, (k, v) in enumerate(importance_relation):
        #     if (k == coalition) and ((v < m) or (M < v)):
        #         print(importance_relation, flush=True)

        return (m, M)

    def amp(
        self,
        importance_relation: tuple[tuple[frozenset[int], float], ...],
        value: float,
    ):
        return (
            0.5
            if np.sum(np.array([v for _, v in importance_relation]) == value) - 1
            else 1
        )

    def replace(
        self, sol: FrozenRMPModel, i: int, value: float, rerank=False, bounds=True
    ):
        key = sol.importance_relation[i][0]

        if bounds:
            m, M = self.bounds(sol.importance_relation, key)
            if (value < m) or (M < value):
                return None

        values = [v for _, v in sol.importance_relation]
        values[i] = value
        if rerank:
            values = rankdata(values, "dense").tolist()

        return replace(
            sol,
            importance_relation=tuple(
                (sol.importance_relation[i][0], values[i])
                for i in range(len(sol.importance_relation))
            ),
        )

    def __call__(self, sol: FrozenRMPModel):
        result: list[FrozenRMPModel] = []

        if differences := self.different_preferences(sol):
            assert self.alternatives
            crits = list(range(len(self.alternatives.criteria)))  # pyright: ignore[reportUnknownArgumentType]
            for rel in differences:
                a = self.alternatives.alternatives_values[rel.a]
                b = self.alternatives.alternatives_values[rel.b]

                profile_ind = 0
                profile = sol.profiles[sol.lexicographic_order[profile_ind]]
                coalition_a = frozenset([c for c in crits if a[c].value >= profile[c]])
                coalition_b = frozenset([c for c in crits if b[c].value >= profile[c]])
                coalition_pair = (coalition_a, coalition_b)

                while (coalition_a == coalition_b) and (
                    profile_ind < len(sol.profiles) - 1
                ):
                    profile_ind += 1
                    profile = sol.profiles[sol.lexicographic_order[profile_ind]]
                    coalition_a = frozenset([
                        c for c in crits if a[c].value >= profile[c]
                    ])
                    coalition_b = frozenset([
                        c for c in crits if b[c].value >= profile[c]
                    ])

                if profile_ind < len(sol.profiles) - 1:
                    a, b = coalition_pair
                    bounds_a = self.bounds(sol.importance_relation, a)
                    bounds_b = self.bounds(sol.importance_relation, b)
                    i_a = -1
                    value_a = -1
                    i_b = -1
                    value_b = -1
                    for i, (key, value) in enumerate(sol.importance_relation):
                        if key == a:
                            i_a = i
                            value_a = value
                        if key == b:
                            i_b = i
                            value_b = value

                    if (
                        (bounds_a[0] <= bounds_b[0] <= bounds_a[1])
                        or (bounds_a[0] <= bounds_b[1] <= bounds_a[1])
                        or (bounds_b[0] <= bounds_a[0] <= bounds_b[1])
                        or (bounds_b[0] <= bounds_a[1] <= bounds_b[1])
                    ):
                        if value_a < value_b:
                            if not a <= b:
                                model = sol
                                model = self.replace(
                                    model, i_a, min(value_b, bounds_a[1])
                                )
                                model = self.replace(
                                    model, i_b, max(value_a, bounds_b[0]), True
                                )
                                result.append(model)
                                # if model is None:
                                #     print("A", bounds_a[0], value_a, bounds_a[1], bounds_b[0], value_b, bounds_b[1], flush=True)
                                #     print(sol.importance_relation, flush=True)

                            model = sol
                            value_median = max(
                                min((value_a + value_b) / 2, bounds_a[1]), bounds_b[0]
                            )
                            model = self.replace(model, i_a, value_median)
                            model = self.replace(model, i_b, value_median, True)
                            result.append(model)
                            # if model is None:
                            #     print("B", bounds_a[0], value_a, bounds_a[1], bounds_b[0], value_b, bounds_b[1], value_median, flush=True)
                            #     print(sol.importance_relation, flush=True)

                        elif value_b < value_a:
                            if not b <= a:
                                model = sol
                                model = self.replace(
                                    model, i_a, max(value_b, bounds_a[0])
                                )
                                model = self.replace(
                                    model, i_b, min(value_a, bounds_b[1]), True
                                )
                                result.append(model)
                                # if model is None:
                                #     print("C", bounds_a[0], value_a, bounds_a[1], bounds_b[0], value_b, bounds_b[1], value_median, flush=True)
                                #     print(sol.importance_relation, flush=True)

                            model = sol
                            value_median = min(
                                max((value_a + value_b) / 2, bounds_a[0]), bounds_b[1]
                            )
                            model = self.replace(model, i_a, value_median)
                            model = self.replace(model, i_b, value_median, True)
                            result.append(model)
                            # if model is None:
                            #     print("D", bounds_a[0], value_a, bounds_a[1], bounds_b[0], value_b, bounds_b[1], value_median, flush=True)
                            #     print(sol.importance_relation, flush=True)
                        else:
                            amp = self.amp(sol.importance_relation, value)
                            if res := self.replace(sol, i_a, value - amp, True):
                                result.append(res)
                            if res := self.replace(sol, i_a, value + amp, True):
                                result.append(res)
                            if res := self.replace(sol, i_b, value - amp, True):
                                result.append(res)
                            if res := self.replace(sol, i_b, value + amp, True):
                                result.append(res)

                    else:
                        model = sol
                        value_median = (value_a + value_b) / 2
                        for i, (key, value) in enumerate(sol.importance_relation):
                            if value_a < value_b:
                                if a <= key and value < value_median:
                                    model = self.replace(
                                        model, i, value_median, bounds=False
                                    )
                                if key <= b and value_median < value:
                                    model = self.replace(
                                        model, i, value_median, bounds=False
                                    )
                            else:
                                if b <= key and value < value_median:
                                    model = self.replace(
                                        model, i, value_median, bounds=False
                                    )
                                if key <= a and value_median < value:
                                    model = self.replace(
                                        model,
                                        i,
                                        value_median,
                                        rerank=True,
                                        bounds=False,
                                    )
                        result.append(model)
                        # if model is None:
                        #     print("E", bounds_a[0], value_a, bounds_a[1], bounds_b[0], value_b, bounds_b[1], value_median, flush=True)
                        #     print(sol.importance_relation, flush=True)
        else:
            for i, (key, value) in enumerate(sol.importance_relation):
                amp = self.amp(sol.importance_relation, value)
                if (res := self.replace(sol, i, value - amp, True)) is not None:
                    result.append(res)
                if (res := self.replace(sol, i, value + amp, True)) is not None:
                    result.append(res)

        return result


@dataclass
class NeighborhoodWeight(NeighborhoodModel[FrozenSRMPModel]):
    def __call__(self, sol: FrozenSRMPModel):
        result: list[FrozenSRMPModel] = []

        for crit, increase in product(range(len(sol.weights)), [False, True]):
            result.append(
                replace(
                    sol,
                    weights=(
                        weights_local_change(np.array(sol.weights), crit, increase)
                    ),
                )
            )

        return result


class NeighborhoodLexOrder[T: FrozenRMPModel | FrozenSRMPModel](NeighborhoodModel[T]):
    def __call__(self, sol: T):
        result: list[T] = []

        for i in range(len(sol.lexicographic_order) - 1):
            result.append(
                replace(
                    sol,
                    lexicographic_order=tuple(
                        adjacent_swap(list(sol.lexicographic_order), i)
                    ),
                ),
            )

        return result
