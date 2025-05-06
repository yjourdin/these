import itertools
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np
from mcda.relations import PreferenceStructure
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


class MIPSRMPCollectiveVars(MIPVars):
    w: D[D[LpVariable]]
    p: D[D[D[LpVariable]]]
    delta: D[D[D[D[LpVariable]]]]
    omega: D[D[D[D[LpVariable]]]]
    s: D[D[D[LpVariable]]]
    s_star: D[D[LpVariable]]
    S: LpVariable
    R: D[LpVariable]
    W_abs: D[D[LpVariable]]
    W: LpVariable
    P_abs: D[D[D[LpVariable]]]
    P: LpVariable


@dataclass
class MIPSRMPCollectiveParams(MIPParams):
    A: list[Any]
    M: list[Any]
    c: int
    lexicographic_order: InitVar[Sequence[Sequence[int]]]
    k: int = field(init=False)
    profile_indices: list[int] = field(init=False)
    sigma: list[list[int]] = field(init=False)
    DM: range = field(init=False)
    Models: range = field(init=False)

    def __post_init__(self, lexicographic_order: Sequence[Sequence[int]]):
        self.DM = range(self.c)  # type: ignore
        self.Models = range(self.c + 1)
        self.k = len(lexicographic_order)
        self.profile_indices = list(range(1, self.k + 1))
        self.sigma = [
            [0] + [profile + 1 for profile in lexicographic_order[model]]
            for model in self.Models
        ]


@dataclass
class MIPSRMPCollective(MIP[SRMPModel, MIPSRMPCollectiveVars, MIPSRMPCollectiveParams]):
    alternatives: NormalPerformanceTable
    preference_relations: list[PreferenceStructure]
    indifference_relations: list[PreferenceStructure]
    lexicographic_order: Sequence[Sequence[int]]
    preferences_changed: list[int]
    preference_refused: list[PreferenceStructure]
    indifference_refused: list[PreferenceStructure]
    count_refused: list[int]
    gamma: float = EPSILON
    best_objective: float | None = None
    penalty: float = 0

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        self.params = MIPSRMPCollectiveParams(
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            c=len(self.preference_relations),
            lexicographic_order=self.lexicographic_order,
        )
        # Binary comparisons with preference
        preference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.preference_relations[dm].relations for dm in self.params.DM
                )
            )
            | set(
                itertools.chain.from_iterable(
                    pref_refused.relations for pref_refused in self.preference_refused
                )
            )
        )
        preference_relations_union_indices = range(len(preference_relations_union))
        # Binary comparisons with indifference
        indifference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.indifference_relations[dm].relations for dm in self.params.DM
                )
            )
            | set(
                itertools.chain.from_iterable(
                    indif_refused.relations
                    for indif_refused in self.indifference_refused
                )
            )
        )
        indifference_relations_union_indices = range(len(indifference_relations_union))

        #############
        # Variables #
        #############

        self.vars = MIPSRMPCollectiveVars(
            w=LpVariable.dicts(
                "Weight",
                (self.params.Models, self.params.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            p=LpVariable.dicts(
                "Profile",
                (self.params.Models, self.params.profile_indices, self.params.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            delta=LpVariable.dicts(
                "LocalConcordance",
                (
                    self.params.Models,
                    self.params.A,
                    self.params.profile_indices,
                    self.params.M,
                ),
                cat=LpBinary,
            ),  # type: ignore
            omega=LpVariable.dicts(
                "WeightedLocalConcordance",
                (
                    self.params.Models,
                    self.params.A,
                    self.params.profile_indices,
                    self.params.M,
                ),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            s=LpVariable.dicts(
                "PreferenceRankingVariable",
                (
                    self.params.Models,
                    preference_relations_union_indices,
                    [0] + self.params.profile_indices,
                ),
                cat=LpBinary,
            ),  # type: ignore
            s_star=LpVariable.dicts(
                "IndifferenceRankingVariable",
                (indifference_relations_union_indices, self.params.Models),
                cat=LpBinary,
            ),  # type: ignore
            S=LpVariable("MinimumPreferencesChanges", cat=LpInteger),  # type: ignore
            R=LpVariable.dicts(
                "Inconsistencies", range(len(self.preference_refused)), cat=LpBinary
            ),  # type: ignore
            W_abs=LpVariable.dicts(
                "WeightAbsoluteDifference", (self.params.DM, self.params.M)
            ),  # type: ignore
            W=LpVariable(
                "WeightDistance",
            ),
            P_abs=LpVariable.dicts(
                "ProfileAbsoluteDifference",
                (self.params.DM, self.params.profile_indices, self.params.M),
            ),  # type: ignore
            P=LpVariable("ProfileDistance"),
        )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMinimize)

        self.prob += (
            self.vars["W"]
            + self.vars["P"]
            + self.penalty
            + 2
            * (self.params.k + 1)
            * len(self.params.M)  # type: ignore
            * (
                self.vars["S"]
                + 2
                * (
                    len(preference_relations_union_indices)
                    + len(indifference_relations_union_indices)
                )
                * lpSum(self.vars["R"])
            )
        )

        ###############
        # Constraints #
        ###############

        # Normalized weights
        for model in self.params.Models:
            self.prob += lpSum([self.vars["w"][model][j] for j in self.params.M]) == 1

            for j in self.params.M:
                # Non-zero weights
                # self.prob += self.vars["w"][j] >= self.gamma

                # Constraints on the reference profiles
                # self.prob += self.vars["p"][1][j] >= 0
                # self.prob += self.vars["p"][self.params.k][j] <= 1

                for h in self.params.profile_indices:
                    if h != self.params.k:
                        # Dominance between the reference profiles
                        self.prob += (
                            self.vars["p"][model][h + 1][j]
                            >= self.vars["p"][model][h][j]
                        )

                    for a in self.params.A:
                        # Constraints on the local concordances
                        self.prob += (
                            self.alternatives.cell[a, j] - self.vars["p"][model][h][j]
                            >= self.vars["delta"][model][a][h][j] - 1
                        )
                        self.prob += (
                            self.vars["delta"][model][a][h][j]
                            >= self.alternatives.cell[a, j]
                            - self.vars["p"][model][h][j]
                            + self.gamma
                        )

                        # Constraints on the weighted local concordances
                        self.prob += (
                            self.vars["omega"][model][a][h][j]
                            <= self.vars["w"][model][j]
                        )
                        self.prob += self.vars["omega"][model][a][h][j] >= 0
                        self.prob += (
                            self.vars["omega"][model][a][h][j]
                            <= self.vars["delta"][model][a][h][j]
                        )
                        self.prob += (
                            self.vars["omega"][model][a][h][j]
                            >= self.vars["delta"][model][a][h][j]
                            + self.vars["w"][model][j]
                            - 1
                        )

            # Constraints on the preference ranking varsiables
            for index in preference_relations_union_indices:
                self.prob += (
                    self.vars["s"][model][index][self.params.sigma[model][0]] == 0
                )

                # for h in self.params.profile_indices:
                #     self.prob += (
                #         self.vars["s"][index][self.params.sigma[h]]
                #         >= self.vars["s"][index][self.params.sigma[h - 1]]
                #     )

            for h in self.params.profile_indices:
                # Constraints on the preferences
                for index, relation in enumerate(preference_relations_union):
                    a, b = relation.a, relation.b
                    self.prob += lpSum([
                        self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                        for j in self.params.M
                    ]) >= (
                        lpSum([
                            self.vars["omega"][model][b][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ])
                        + self.gamma
                        - (1 + self.gamma)
                        * (
                            1
                            - self.vars["s"][model][index][self.params.sigma[model][h]]
                            + self.vars["s"][model][index][
                                self.params.sigma[model][h - 1]
                            ]
                        )
                    )

                    self.prob += lpSum([
                        self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                        for j in self.params.M
                    ]) >= (
                        lpSum([
                            self.vars["omega"][model][b][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ])
                        - self.vars["s"][model][index][self.params.sigma[model][h]]
                        - self.vars["s"][model][index][self.params.sigma[model][h - 1]]
                    )

                    self.prob += lpSum([
                        self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                        for j in self.params.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][model][b][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ])
                        + self.vars["s"][model][index][self.params.sigma[model][h]]
                        + self.vars["s"][model][index][self.params.sigma[model][h - 1]]
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(indifference_relations_union):
                    a, b = relation.a, relation.b
                    if model == self.params.c:
                        self.prob += lpSum([
                            self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ]) >= (
                            lpSum([
                                self.vars["omega"][model][b][
                                    self.params.sigma[model][h]
                                ][j]
                                for j in self.params.M
                            ])
                            - self.vars["s_star"][model][index]
                        )

                        self.prob += lpSum([
                            self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ]) <= (
                            lpSum([
                                self.vars["omega"][model][b][
                                    self.params.sigma[model][h]
                                ][j]
                                for j in self.params.M
                            ])
                            + self.vars["s_star"][model][index]
                        )
                    else:
                        self.prob += lpSum([
                            self.vars["omega"][model][a][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ]) == lpSum([
                            self.vars["omega"][model][b][self.params.sigma[model][h]][j]
                            for j in self.params.M
                        ])

                    # self.prob += lpSum(
                    #     [
                    #         self.vars["omega"][a][self.params.sigma[h]][j]
                    #         for j in self.params.M
                    #     ]
                    # ) >= (
                    #     lpSum(
                    #         [
                    #             self.vars["omega"][b][self.params.sigma[h]][j]
                    #             for j in self.params.M
                    #         ]
                    #     )
                    #     + self.gamma * self.vars["s_star"][index]
                    # )

                    # self.prob += lpSum(
                    #     [
                    #         self.vars["omega"][b][self.params.sigma[h]][j]
                    #         for j in self.params.M
                    #     ]
                    # ) >= (
                    #     lpSum(
                    #         [
                    #             self.vars["omega"][a][self.params.sigma[h]][j]
                    #             for j in self.params.M
                    #         ]
                    #     )
                    #     + self.gamma * self.vars["s_star"][index]
                    # )

        # Constraint on refused preferences
        for index in range(len(self.preference_refused)):
            self.prob += lpSum([
                self.vars["s"][self.params.c][preference_relations_union.index(r)][
                    self.params.sigma[self.params.c][self.params.k]
                ]
                for r in self.preference_refused[index]
            ]) + lpSum([
                1
                - self.vars["s_star"][self.params.c][
                    indifference_relations_union.index(r)
                ]
                for r in self.indifference_refused[index]
            ]) <= len(
                set(frozenset(r.elements) for r in self.preference_refused[index])
                | set(frozenset(r.elements) for r in self.indifference_refused[index])
            ) - self.count_refused[index] * (1 - self.vars["R"][index])

        # Constraints on minimum number of preferences changes
        for dm in self.params.DM:
            self.prob += self.vars["S"] >= self.preferences_changed[dm] + lpSum([
                1
                - self.vars["s"][self.params.c][preference_relations_union.index(r)][
                    self.params.sigma[self.params.c][self.params.k]
                ]
                for r in self.preference_relations[dm]
            ]) + lpSum([
                self.vars["s_star"][self.params.c][
                    indifference_relations_union.index(r)
                ]
                for r in self.indifference_relations[dm]
            ])

        # Distance
        for dm in self.params.DM:
            for j in self.params.M:
                self.prob += (
                    self.vars["W_abs"][dm][j]
                    >= self.vars["w"][dm][j] - self.vars["w"][self.params.c][j]
                )
                self.prob += (
                    self.vars["W_abs"][dm][j]
                    >= self.vars["w"][self.params.c][j] - self.vars["w"][dm][j]
                )

                for h in self.params.profile_indices:
                    self.prob += (
                        self.vars["P_abs"][dm][h][j]
                        >= self.vars["p"][dm][h][j]
                        - self.vars["p"][self.params.c][h][j]
                    )
                    self.prob += (
                        self.vars["P_abs"][dm][h][j]
                        >= self.vars["p"][self.params.c][h][j]
                        - self.vars["p"][dm][h][j]
                    )

            self.prob += self.vars["W"] >= lpSum([
                self.vars["W_abs"][dm][j] for j in self.params.M
            ])
            self.prob += self.vars["P"] >= lpSum([
                self.vars["P_abs"][dm][h][j]
                for j in self.params.M
                for h in self.params.profile_indices
            ])

        if self.best_objective is not None:
            self.prob += (
                self.vars["S"]
                + 2
                * (
                    len(preference_relations_union_indices)
                    + len(indifference_relations_union_indices)
                )
                * lpSum(self.vars["R"])
                <= self.best_objective - 1
            )

    def create_solution(self):
        weights = np.array([
            cast(float, value(self.vars["w"][self.params.c][j])) for j in self.params.M
        ])
        profiles = NormalPerformanceTable([
            [value(self.vars["p"][self.params.c][h][j]) for j in self.params.M]
            for h in self.params.profile_indices
        ])

        return SRMPModel(
            profiles=profiles,
            weights=weights,
            lexicographic_order=[p - 1 for p in self.params.sigma[self.params.c][1:]],
        )
