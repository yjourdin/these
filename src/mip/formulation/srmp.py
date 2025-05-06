from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np
from mcda.relations import I, P
from pulp import (  # type: ignore
    LpBinary,
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from ...constants import EPSILON
from ...performance_table.normal_performance_table import NormalPerformanceTable
from ...srmp.model import SRMPModel
from ..mip import MIP, D, MIPParams, MIPVars


class MIPSRMPVars(MIPVars):
    w: D[LpVariable]
    p: D[D[LpVariable]]
    delta: D[D[D[LpVariable]]]
    omega: D[D[D[LpVariable]]]
    s: D[D[LpVariable]]
    s_star: D[LpVariable]


@dataclass
class MIPSRMPParams(MIPParams):
    A: list[Any]
    M: list[Any]
    lexicographic_order: InitVar[Sequence[int]]
    k: int = field(init=False)
    profile_indices: list[int] = field(init=False)
    sigma: list[int] = field(init=False)

    def __post_init__(self, lexicographic_order: Sequence[int]):
        self.k = len(lexicographic_order)
        self.profile_indices = list(range(1, self.k + 1))
        self.sigma = [0] + [profile + 1 for profile in lexicographic_order]


@dataclass
class MIPSRMP(MIP[SRMPModel, MIPSRMPVars, MIPSRMPParams]):
    alternatives: NormalPerformanceTable
    preference_relations: list[P]
    indifference_relations: list[I]
    lexicographic_order: Sequence[int]
    gamma: float = EPSILON
    inconsistencies: bool = True
    best_fitness: float | None = None

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        self.param = MIPSRMPParams(
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            lexicographic_order=self.lexicographic_order,
        )
        # Binary comparisons with preference
        preference_relations_indices = range(len(self.preference_relations))
        # Binary comparisons with indifference
        indifference_relations_indices = range(len(self.indifference_relations))

        #############
        # Variables #
        #############

        self.vars = MIPSRMPVars(
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
                (preference_relations_indices, [0] + self.param.profile_indices),
                cat=LpBinary,
            ),  # type: ignore
            s_star=(
                LpVariable.dicts(
                    "IndifferenceRankingVariable",
                    indifference_relations_indices,
                    cat=LpBinary,
                )
                if self.inconsistencies
                else {}  # type: ignore
            ),
        )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum([
                self.vars["s"][index][0] for index in preference_relations_indices
            ]) + lpSum([
                self.vars["s_star"][index] for index in indifference_relations_indices
            ])

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
        for index in preference_relations_indices:
            if not self.inconsistencies:
                self.prob += self.vars["s"][index][self.param.sigma[0]] == 1
            self.prob += self.vars["s"][index][self.param.sigma[self.param.k]] == 0

        for h in self.param.profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(self.preference_relations):
                a, b = relation.a, relation.b

                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    + self.gamma
                    - self.vars["s"][index][self.param.sigma[h]] * (1 + self.gamma)
                    - (1 - self.vars["s"][index][self.param.sigma[h - 1]])
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    - (1 - self.vars["s"][index][self.param.sigma[h]])
                    - (1 - self.vars["s"][index][self.param.sigma[h - 1]])
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.param.sigma[h]][j] for j in self.param.M
                ]) <= (
                    lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                    + (1 - self.vars["s"][index][self.param.sigma[h]])
                    + (1 - self.vars["s"][index][self.param.sigma[h - 1]])
                )

            # Constraints on the indifferences
            for index, relation in enumerate(self.indifference_relations):
                a, b = relation.a, relation.b
                if not self.inconsistencies:
                    self.prob += lpSum([
                        self.vars["omega"][a][self.param.sigma[h]][j]
                        for j in self.param.M
                    ]) == lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ])
                else:
                    self.prob += lpSum([
                        self.vars["omega"][a][self.param.sigma[h]][j]
                        for j in self.param.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][b][self.param.sigma[h]][j]
                            for j in self.param.M
                        ])
                        - (1 - self.vars["s_star"][index])
                    )

                    self.prob += lpSum([
                        self.vars["omega"][b][self.param.sigma[h]][j]
                        for j in self.param.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][a][self.param.sigma[h]][j]
                            for j in self.param.M
                        ])
                        - (1 - self.vars["s_star"][index])
                    )

        if self.inconsistencies and (self.best_fitness is not None):
            self.prob += (
                lpSum([
                    self.vars["s"][index][0] for index in preference_relations_indices
                ])
                + lpSum([
                    self.vars["s_star"][index]
                    for index in indifference_relations_indices
                ])
                >= self.best_fitness + self.gamma
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
