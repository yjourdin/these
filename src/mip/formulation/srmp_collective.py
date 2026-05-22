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
)

from src.constants import EPSILON
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.srmp.model import SRMPModel

from ..mip import MIP, D, MIPParams, MIPVars, value


class MIPSRMPCollectiveVars(MIPVars):
    w: D[LpVariable]
    p: D[D[LpVariable]]
    delta: D[D[D[LpVariable]]]
    omega: D[D[D[LpVariable]]]
    s: D[D[LpVariable]]
    s_star: D[LpVariable]
    S: LpVariable


@dataclass
class MIPSRMPCollectiveParams(MIPParams):
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


@dataclass(kw_only=True)
class MIPSRMPCollective(MIP[SRMPModel, MIPSRMPCollectiveVars, MIPSRMPCollectiveParams]):
    alternatives: NormalPerformanceTable
    preference_relations: list[list[P]]
    indifference_relations: list[list[I]]
    lexicographic_order: Sequence[int]
    preferences_changed: list[int]
    preference_refused: list[P]
    indifference_refused: list[I]
    preference_accepted: list[P]
    indifference_accepted: list[I]
    gamma: float = EPSILON
    best_objective: float | None = None

    def create_parameters(self):
        self.params = MIPSRMPCollectiveParams(
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            DM=range(len(self.preference_relations)),
            lexicographic_order=self.lexicographic_order,
        )

    def create_variables(self):
        # Binary comparisons with preference
        self.preference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.preference_relations[dm] for dm in self.params.DM
                )
            )
            | set(self.preference_accepted)
            | {P(r.b, r.a) for r in self.preference_refused}
            | {P(r.a, r.b) for r in self.indifference_refused}
            | {P(r.b, r.a) for r in self.indifference_refused}
        )
        preference_relations_union_indices = range(len(self.preference_relations_union))
        # Binary comparisons with indifference
        self.indifference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.indifference_relations[dm] for dm in self.params.DM
                )
            )
            | set(self.indifference_accepted)
            | {I(r.a, r.b) for r in self.preference_refused}
        )
        indifference_relations_union_indices = range(
            len(self.indifference_relations_union)
        )

        #############
        # Variables #
        #############

        self.vars = MIPSRMPCollectiveVars(
            w=LpVariable.dicts("Weight", self.params.M, lowBound=0, upBound=1),  # type: ignore
            p=LpVariable.dicts(
                "Profile",
                (self.params.profile_indices, self.params.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            delta=LpVariable.dicts(
                "LocalConcordance",
                (self.params.A, self.params.profile_indices, self.params.M),
                cat=LpBinary,
            ),  # type: ignore
            omega=LpVariable.dicts(
                "WeightedLocalConcordance",
                (self.params.A, self.params.profile_indices, self.params.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            s=LpVariable.dicts(
                "PreferenceRankingVariable",
                (preference_relations_union_indices, [0] + self.params.profile_indices),
                cat=LpBinary,
            ),  # type: ignore
            s_star=LpVariable.dicts(
                "IndifferenceRankingVariable",
                indifference_relations_union_indices,
                cat=LpBinary,
            ),  # type: ignore
            S=LpVariable("MinimumPreferencesChanges", cat=LpInteger),
        )

    def create_problem(self):
        self.prob = LpProblem("SRMP_Elicitation", LpMinimize)

        self.prob += self.vars["S"]
        # self.prob += self.vars["S"] * max(len(self.preference_relations[dm]) + len(self.indifference_relations[dm]) for dm in self.params.DM) + (
        #     lpSum([self.preferences_changed[dm] for dm in self.params.DM])
        #     + sum(
        #         lpSum([
        #             self.vars["s"][preference_relations_union.index(r)][0]
        #             for r in self.preference_relations[dm]
        #         ])
        #         for dm in self.params.DM
        #     )
        #     + sum(
        #         lpSum([
        #             self.vars["s_star"][indifference_relations_union.index(r)]
        #             for r in self.indifference_relations[dm]
        #         ])
        #         for dm in self.params.DM
        #     )
        # ) / len(self.params.DM)

        # Normalized weights
        self.prob += lpSum([self.vars["w"][j] for j in self.params.M]) == 1

        for j in self.params.M:
            # Non-zero weights
            # self.prob += self.vars["w"][j] >= self.gamma

            # Constraints on the reference profiles
            # self.prob += self.vars["p"][1][j] >= 0
            # self.prob += self.vars["p"][self.params.k][j] <= 1

            for h in self.params.profile_indices:
                if h != self.params.k:
                    # Dominance between the reference profiles
                    self.prob += self.vars["p"][h + 1][j] >= self.vars["p"][h][j]

                for a in self.params.A:
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
                    # self.prob += self.vars["omega"][a][h][j] >= 0
                    self.prob += (
                        self.vars["omega"][a][h][j] <= self.vars["delta"][a][h][j]
                    )
                    self.prob += (
                        self.vars["omega"][a][h][j]
                        >= self.vars["delta"][a][h][j] + self.vars["w"][j] - 1
                    )

                    # if h > 1:
                    #     self.prob += self.vars["omega"][a][h][j] <= self.vars["omega"][a][h-1][j]

        # Constraints on the preference ranking variables
        for s in self.vars["s"].values():
            self.prob += s[self.params.sigma[self.params.k]] == 1

            # for h in self.params.profile_indices:
            #     self.prob += s[self.params.sigma[h - 1]] <= s[self.params.sigma[h]]

        for h in self.params.profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(self.preference_relations_union):
                a, b = relation.a, relation.b
                self.prob += lpSum([
                    self.vars["omega"][a][self.params.sigma[h]][j]
                    for j in self.params.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    + self.gamma
                    - (1 + self.gamma)
                    * (
                        1
                        - self.vars["s"][index][self.params.sigma[h]]
                        + self.vars["s"][index][self.params.sigma[h - 1]]
                    )
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.params.sigma[h]][j]
                    for j in self.params.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    - self.vars["s"][index][self.params.sigma[h]]
                    - self.vars["s"][index][self.params.sigma[h - 1]]
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.params.sigma[h]][j]
                    for j in self.params.M
                ]) <= (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    + self.vars["s"][index][self.params.sigma[h]]
                    + self.vars["s"][index][self.params.sigma[h - 1]]
                )

            # Constraints on the indifferences
            for index, relation in enumerate(self.indifference_relations_union):
                a, b = relation.a, relation.b
                self.prob += (
                    lpSum([
                        self.vars["omega"][a][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    - lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    <= self.vars["s_star"][index]
                )

                self.prob += (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    - lpSum([
                        self.vars["omega"][a][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    <= self.vars["s_star"][index]
                )

        # Constraint on accepted preferences
        # for r in self.preference_accepted:
        #     self.vars["s"][self.preference_relations_union.index(r)][0].setInitialValue(
        #         0
        #     )
        #     self.vars["s"][self.preference_relations_union.index(r)][0].fixValue()

        # for r in self.indifference_accepted:
        #     self.vars["s_star"][
        #         self.indifference_relations_union.index(r)
        #     ].setInitialValue(0)
        #     self.vars["s_star"][self.indifference_relations_union.index(r)].fixValue()

        # Constraint on refused preferences
        for r in self.preference_refused:
            self.prob += (
                self.vars["s"][self.preference_relations_union.index(P(r.b, r.a))][0]
                + self.vars["s_star"][
                    self.indifference_relations_union.index(I(r.a, r.b))
                ]
                <= 1
            )

        for r in self.indifference_refused:
            self.prob += (
                self.vars["s"][self.preference_relations_union.index(P(r.a, r.b))][0]
                + self.vars["s"][self.preference_relations_union.index(P(r.b, r.a))][0]
                <= 1
            )

        # Constraints on minimum number of preferences changes
        for dm in self.params.DM:
            self.prob += self.vars["S"] >= self.preferences_changed[dm] + lpSum([
                self.vars["s"][self.preference_relations_union.index(r)][0]
                for r in self.preference_relations[dm]
            ]) + lpSum([
                self.vars["s_star"][self.indifference_relations_union.index(r)]
                for r in self.indifference_relations[dm]
            ])

        if self.best_objective is not None:
            self.prob += self.vars["S"] <= self.best_objective - 1

    def create_solution(self):
        weights = np.array([
            cast(float, value(self.vars["w"][j])) for j in self.params.M
        ])
        profiles = NormalPerformanceTable([
            [value(self.vars["p"][h][j]) for j in self.params.M]
            for h in self.params.profile_indices
        ])

        self.sol = SRMPModel(
            profiles=profiles,
            weights=weights,
            lexicographic_order=[p - 1 for p in self.params.sigma[1:]],
        )
