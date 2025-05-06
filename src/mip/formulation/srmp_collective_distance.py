import itertools
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np
from mcda.relations import I, P
from pulp import (  # type: ignore
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from ...constants import EPSILON
from ...performance_table.normal_performance_table import NormalPerformanceTable
from ...srmp.model import SRMPModel
from ..mip import MIP, D, MIPParams, MIPVars


class MIPSRMPCollectiveDistanceVars(MIPVars):
    w: D[LpVariable]
    p: D[D[LpVariable]]
    delta: D[D[D[LpVariable]]]
    omega: D[D[D[LpVariable]]]
    s: D[D[LpVariable]]
    s_star: D[LpVariable]
    S: LpVariable


@dataclass
class MIPSRMPCollectiveDistanceParams(MIPParams):
    A: list[Any]
    M: list[Any]
    DM: range
    lexicographic_order: InitVar[Sequence[int]]
    k: int = field(init=False)
    profile_indices: list[int] = field(init=False)
    sigma: list[int] = field(init=False)

    def __post_init__(self, lexicographic_order: Sequence[int]):
        self.k = len(lexicographic_order)
        self.profile_indices = list(range(1, self.k + 1))
        self.sigma = [0] + [profile + 1 for profile in lexicographic_order]


@dataclass
class MIPSRMPCollectiveDistance(
    MIP[SRMPModel, MIPSRMPCollectiveDistanceVars, MIPSRMPCollectiveDistanceParams]
):
    alternatives: NormalPerformanceTable
    preference_relations: list[list[P]]
    indifference_relations: list[list[I]]
    lexicographic_order: Sequence[int]
    preferences_changed: list[int]
    preference_to_accept: list[list[P]]
    indifference_to_accept: list[list[I]]
    preference_accepted: list[P]
    indifference_accepted: list[I]
    model: SRMPModel
    profiles_amp: float
    weights_amp: float
    gamma: float = EPSILON
    best_objective: float | None = None

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        self.param = MIPSRMPCollectiveDistanceParams(
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            DM=range(len(self.preference_relations)),
            lexicographic_order=self.lexicographic_order,
        )
        # Binary comparisons with preference
        preference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.preference_relations[dm] for dm in self.param.DM
                )
            )
            | set(
                itertools.chain.from_iterable(
                    pref_refused for pref_refused in self.preference_to_accept
                )
            )
            | set(self.preference_accepted)
        )
        preference_relations_union_indices = range(len(preference_relations_union))
        # Binary comparisons with indifference
        indifference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.indifference_relations[dm] for dm in self.param.DM
                )
            )
            | set(
                itertools.chain.from_iterable(
                    indif_refused for indif_refused in self.indifference_to_accept
                )
            )
            | set(self.indifference_accepted)
        )
        indifference_relations_union_indices = range(len(indifference_relations_union))

        #############
        # Variables #
        #############

        self.vars = MIPSRMPCollectiveDistanceVars(
            w=LpVariable.dicts("Weight", self.param.M, lowBound=0, upBound=1),  # type: ignore
            p=LpVariable.dicts(
                "Profile",
                (self.param.profile_indices, self.param.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            delta=LpVariable.dicts(
                "LocalConcordance",
                (self.param.A, self.param.profile_indices, self.param.M),
                cat=LpBinary,
            ),  # type: ignore
            omega=LpVariable.dicts(
                "WeightedLocalConcordance",
                (self.param.A, self.param.profile_indices, self.param.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            s=LpVariable.dicts(
                "PreferenceRankingVariable",
                (preference_relations_union_indices, [0] + self.param.profile_indices),
                cat=LpBinary,
            ),  # type: ignore
            s_star=LpVariable.dicts(
                "IndifferenceRankingVariable",
                indifference_relations_union_indices,
                cat=LpBinary,
            ),  # type: ignore
            S=LpVariable("MinimumPreferencesChanges", cat=LpInteger),
        )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMinimize)

        self.prob += self.vars["S"]

        ###############
        # Constraints #
        ###############

        # Normalized weights
        self.prob += lpSum([self.vars["w"][j] for j in self.param.M]) == 1

        for j in self.param.M:
            # Non-zero weights
            # self.prob += self.vars["w"][j] >= self.gamma

            # Constraints on the reference profiles
            # self.prob += self.vars["p"][1][j] >= 0
            # self.prob += self.vars["p"][self.param.k][j] <= 1

            for h in self.param.profile_indices:
                if h != self.param.k:
                    # Dominance between the reference profiles
                    self.prob += self.vars["p"][h + 1][j] >= self.vars["p"][h][j]

                for a in self.param.A:
                    # Constraints on the local concordances
                    self.prob += (
                        self.alternatives.cell[a, j] - self.vars["p"][h][j]
                        >= self.vars["delta"][a][h][j] - 1
                    )
                    self.prob += (
                        self.vars["delta"][a][h][j]
                        >= self.alternatives.cell[a, j]
                        - self.vars["p"][h][j]
                        + self.gamma
                    )

                    # Constraints on the weighted local concordances
                    self.prob += self.vars["omega"][a][h][j] <= self.vars["w"][j]
                    self.prob += self.vars["omega"][a][h][j] >= 0
                    self.prob += (
                        self.vars["omega"][a][h][j] <= self.vars["delta"][a][h][j]
                    )
                    self.prob += (
                        self.vars["omega"][a][h][j]
                        >= self.vars["delta"][a][h][j] + self.vars["w"][j] - 1
                    )

        # Constraints on the preference ranking variables
        for index in preference_relations_union_indices:
            self.prob += self.vars["s"][index][self.param.sigma[self.param.k]] == 1

        for h in self.param.profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(preference_relations_union):
                a, b = relation.a, relation.b
                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    + self.gamma
                    - (1 + self.gamma)
                    * (
                        1
                        - self.vars["s"][index][self.param.sigma[h]]
                        + self.vars["s"][index][self.param.sigma[h - 1]]
                    )
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    - self.vars["s"][index][self.param.sigma[h]]
                    - self.vars["s"][index][self.param.sigma[h - 1]]
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) <= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    + self.vars["s"][index][self.param.sigma[h]]
                    + self.vars["s"][index][self.param.sigma[h - 1]]
                )

            # Constraints on the indifferences
            for index, relation in enumerate(indifference_relations_union):
                a, b = relation.a, relation.b
                self.prob += (
                    lpSum([
                        self.vars["omega"][a][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    - lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    <= self.vars["s_star"][index]
                )

                self.prob += (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    - lpSum([
                        self.vars["omega"][a][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    <= self.vars["s_star"][index]
                )

        # Constraint on refused preferences
        for index in range(len(self.preference_to_accept)):
            self.prob += (
                lpSum([
                    self.vars["s"][preference_relations_union.index(r)][0]
                    for r in self.preference_to_accept[index]
                ])
                + lpSum([
                    self.vars["s_star"][indifference_relations_union.index(r)]
                    for r in self.indifference_to_accept[index]
                ])
                <= len(self.preference_to_accept[index])
                + len(self.indifference_to_accept[index])
                - 1
            )

        # Constraint on accepted preferences
        self.prob += (
            lpSum([
                self.vars["s"][preference_relations_union.index(r)][0]
                for r in self.preference_accepted
            ])
            + lpSum([
                self.vars["s_star"][indifference_relations_union.index(r)]
                for r in self.indifference_accepted
            ])
            <= 0
        )

        # Constraints on minimum number of preferences changes
        for dm in self.param.DM:
            self.prob += self.vars["S"] >= self.preferences_changed[dm] + lpSum([
                self.vars["s"][preference_relations_union.index(r)][0]
                for r in self.preference_relations[dm]
            ]) + lpSum([
                self.vars["s_star"][indifference_relations_union.index(r)]
                for r in self.indifference_relations[dm]
            ])

        if self.best_objective is not None:
            self.prob += self.vars["S"] <= self.best_objective - 1
        
        # Constraints to accept
        for j in self.params.M:
            self.prob += (
                self.vars["w"][j] >= self.model.weights[j] - self.weights_amp
            )
            self.prob += (
                self.vars["w"][j] <= self.model.weights[j] + self.weights_amp
            )

            for h in self.params.profile_indices:
                self.prob += (
                    self.vars["p"][h][j]
                    >= self.model.profiles.cell[h - 1, j] - self.profiles_amp
                )
                self.prob += (
                    self.vars["p"][h][j]
                    <= self.model.profiles.cell[h - 1, j] + self.profiles_amp
                )

    def create_solution(self):
        weights = np.array([
            cast(float, value(self.vars["w"][j])) for j in self.param.M
        ])
        profiles = NormalPerformanceTable([
            [value(self.vars["p"][h][j]) for j in self.param.M]
            for h in self.param.profile_indices
        ])

        return SRMPModel(
            profiles=profiles,
            weights=weights,
            lexicographic_order=[p - 1 for p in self.param.sigma[1:]],
        )
