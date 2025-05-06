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
from ...srmp.model import (
    SRMPGroupModel,
    SRMPGroupModelProfiles,
    SRMPGroupModelWeights,
    SRMPGroupModelWeightsProfiles,
    SRMPParamFlag,
    srmp_group_model,
)
from ..mip import MIP, D, MIPParams, MIPVars


class MIPSRMPGroupVars(MIPVars):
    w: D[D[LpVariable]]
    p: D[D[D[LpVariable]]]
    delta: D[D[D[D[LpVariable]]]]
    omega: D[D[D[D[LpVariable]]]]
    s: D[D[D[LpVariable]]]
    s_star: D[D[LpVariable]]


@dataclass
class MIPSRMPGroupParams(MIPParams):
    profiles_shared: bool
    weights_shared: bool
    A: list[Any]
    M: list[Any]
    DM: range
    lexicographic_order: InitVar[Sequence[Sequence[int]]]
    k: int = field(init=False)
    profile_indices: list[int] = field(init=False)
    sigma: list[list[int]] = field(init=False)

    def __post_init__(self, lexicographic_order: Sequence[Sequence[int]]):
        self.k = len(lexicographic_order)
        self.profile_indices = list(range(1, self.k + 1))
        self.sigma = [
            [0] + [profile + 1 for profile in lexicographic_order[dm]] for dm in self.DM
        ]


@dataclass
class MIPSRMPGroup(
    MIP[
        SRMPGroupModelWeightsProfiles
        | SRMPGroupModelWeights
        | SRMPGroupModelProfiles
        | SRMPGroupModel,
        MIPSRMPGroupVars,
        MIPSRMPGroupParams,
    ]
):
    alternatives: NormalPerformanceTable
    preference_relations: list[list[P]]
    indifference_relations: list[list[I]]
    lexicographic_order: Sequence[Sequence[int]]
    shared_params: SRMPParamFlag = SRMPParamFlag(0)
    gamma: float = EPSILON
    inconsistencies: bool = True
    best_fitness: float | None = None

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        self.param = MIPSRMPGroupParams(
            profiles_shared=SRMPParamFlag.PROFILES in self.shared_params,
            weights_shared=SRMPParamFlag.WEIGHTS in self.shared_params,
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            DM=range(len(self.preference_relations)),
            lexicographic_order=self.lexicographic_order,
        )
        # Binary comparisons with preference
        preference_relations_indices = [
            range(len(self.preference_relations[dm])) for dm in self.params.DM
        ]
        # Binary comparisons with indifference
        indifference_relations_indices = [
            range(len(self.indifference_relations[dm])) for dm in self.params.DM
        ]

        #############
        # Variables #
        #############

        self.vars = MIPSRMPGroupVars(
            w=LpVariable.dicts(
                "Weight", (self.params.DM, self.params.M), lowBound=0, upBound=1
            ),  # type: ignore
            p=LpVariable.dicts(
                "Profile",
                (self.params.DM, self.params.profile_indices, self.params.M),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            delta=LpVariable.dicts(
                "LocalConcordance",
                (
                    self.params.DM,
                    self.params.A,
                    self.params.profile_indices,
                    self.params.M,
                ),
                cat=LpBinary,
            ),  # type: ignore
            omega=LpVariable.dicts(
                "WeightedLocalConcordance",
                (
                    self.params.DM,
                    self.params.A,
                    self.params.profile_indices,
                    self.params.M,
                ),
                lowBound=0,
                upBound=1,
            ),  # type: ignore
            s={
                dm: LpVariable.dicts(
                    f"PreferenceRankingVariable_{dm}",
                    (
                        preference_relations_indices[dm],
                        [0] + self.params.profile_indices,
                    ),
                    cat=LpBinary,
                )
                for dm in self.params.DM
            },
            s_star=(
                {
                    dm: LpVariable.dicts(
                        f"IndifferenceRankingVariable_{dm}",
                        indifference_relations_indices[dm],
                        cat=LpBinary,
                    )
                    for dm in self.params.DM
                }
                if self.inconsistencies
                else {}
            ),
        )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum([
                [
                    self.vars["s"][dm][index][0]
                    for index in preference_relations_indices[dm]
                ]
                for dm in self.params.DM
            ]) + lpSum([
                [
                    self.vars["s_star"][dm][index]
                    for index in indifference_relations_indices[dm]
                ]
                for dm in self.params.DM
            ])

        ###############
        # Constraints #
        ###############

        # Normalized weights
        for dm in self.params.DM:
            self.prob += lpSum([self.vars["w"][dm][j] for j in self.params.M]) == 1

        for j in self.params.M:
            # for dm in self.params.DM:
            # Non-zero weights
            # self.prob += self.vars["w"][dm][j] >= self.gamma

            # for dm in self.params.DM:
            # Constraints on the reference profiles
            # self.prob += self.vars["p"][dm][1][j] >= 0
            # self.prob += self.vars["p"][dm][self.params.k][j] <= 1

            for h in self.params.profile_indices:
                if h != self.params.k:
                    for dm in self.params.DM:
                        # Dominance between the reference profiles
                        self.prob += (
                            self.vars["p"][dm][h + 1][j] >= self.vars["p"][dm][h][j]
                        )

                for a in self.params.A:
                    for dm in self.params.DM:
                        # Constraints on the local concordances
                        self.prob += (
                            self.alternatives.cell[a, j] - self.vars["p"][dm][h][j]
                            >= self.vars["delta"][dm][a][h][j] - 1
                        )
                        self.prob += (
                            self.vars["delta"][dm][a][h][j]
                            >= self.alternatives.cell[a, j]
                            - self.vars["p"][dm][h][j]
                            + self.gamma
                        )

                    for dm in self.params.DM:
                        # Constraints on the weighted local concordances
                        self.prob += (
                            self.vars["omega"][dm][a][h][j] <= self.vars["w"][dm][j]
                        )
                        self.prob += self.vars["omega"][dm][a][h][j] >= 0
                        self.prob += (
                            self.vars["omega"][dm][a][h][j]
                            <= self.vars["delta"][dm][a][h][j]
                        )
                        self.prob += (
                            self.vars["omega"][dm][a][h][j]
                            >= self.vars["delta"][dm][a][h][j]
                            + self.vars["w"][dm][j]
                            - 1
                        )

        # Constraints on the preference ranking varsiables
        for dm in self.params.DM:
            for index in preference_relations_indices[dm]:
                if not self.inconsistencies:
                    self.prob += (
                        self.vars["s"][dm][index][self.params.sigma[dm][0]] == 1
                    )
                self.prob += (
                    self.vars["s"][dm][index][self.params.sigma[dm][self.params.k]] == 0
                )

        for h in self.params.profile_indices:
            # Constraints on the preferences
            for dm in self.params.DM:
                for index, relation in enumerate(self.preference_relations[dm]):
                    a, b = relation.a, relation.b
                    self.prob += lpSum([
                        self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                        for j in self.params.M
                    ]) >= (
                        lpSum([
                            self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ])
                        + self.gamma
                        - self.vars["s"][dm][index][self.params.sigma[dm][h]]
                        * (1 + self.gamma)
                        - (1 - self.vars["s"][dm][index][self.params.sigma[dm][h - 1]])
                    )

                    self.prob += lpSum([
                        self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                        for j in self.params.M
                    ]) >= (
                        lpSum([
                            self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ])
                        - (1 - self.vars["s"][dm][index][self.params.sigma[dm][h]])
                        - (1 - self.vars["s"][dm][index][self.params.sigma[dm][h - 1]])
                    )

                    self.prob += lpSum([
                        self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                        for j in self.params.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ])
                        + (1 - self.vars["s"][dm][index][self.params.sigma[dm][h]])
                        + (1 - self.vars["s"][dm][index][self.params.sigma[dm][h - 1]])
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(self.indifference_relations[dm]):
                    a, b = relation.a, relation.b
                    if not self.inconsistencies:
                        self.prob += lpSum([
                            self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ]) == lpSum([
                            self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ])
                    else:
                        self.prob += lpSum([
                            self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ]) <= (
                            lpSum([
                                self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                                for j in self.params.M
                            ])
                            - (1 - self.vars["s_star"][dm][index])
                        )

                        self.prob += lpSum([
                            self.vars["omega"][dm][b][self.params.sigma[dm][h]][j]
                            for j in self.params.M
                        ]) <= (
                            lpSum([
                                self.vars["omega"][dm][a][self.params.sigma[dm][h]][j]
                                for j in self.params.M
                            ])
                            - (1 - self.vars["s_star"][dm][index])
                        )

        # Constraint on shared paramseters
        if self.params.profiles_shared or self.params.weights_shared:
            for j in self.params.M:
                for dm in self.params.DM[:-1]:
                    if self.params.weights_shared:
                        self.prob += self.vars["w"][dm][j] == self.vars["w"][dm + 1][j]
                    if self.params.profiles_shared:
                        for h in self.params.profile_indices:
                            self.prob += (
                                self.vars["p"][dm][h][j] == self.vars["p"][dm + 1][h][j]
                            )

        if self.inconsistencies and (self.best_fitness is not None):
            self.prob += (
                lpSum([
                    [
                        self.vars["s"][dm][index][0]
                        for index in preference_relations_indices[dm]
                    ]
                    for dm in self.params.DM
                ])
                + lpSum([
                    [
                        self.vars["s_star"][dm][index]
                        for index in indifference_relations_indices[dm]
                    ]
                    for dm in self.params.DM
                ])
                >= self.best_fitness + self.gamma
            )

    def create_solution(self):
        weights = (
            np.array([cast(float, value(self.vars["w"][0][j])) for j in self.params.M])
            if self.params.weights_shared
            else [
                np.array([
                    cast(float, value(self.vars["w"][dm][j])) for j in self.params.M
                ])
                for dm in self.params.DM
            ]
        )
        profiles = (
            NormalPerformanceTable([
                [value(self.vars["p"][0][h][j]) for j in self.params.M]
                for h in self.params.profile_indices
            ])
            if self.params.profiles_shared
            else [
                NormalPerformanceTable([
                    [value(self.vars["p"][dm][h][j]) for j in self.params.M]
                    for h in self.params.profile_indices
                ])
                for dm in self.params.DM
            ]
        )

        return srmp_group_model(self.shared_params)(
            group_size=len(self.params.DM),
            profiles=profiles,
            weights=weights,
            lexicographic_order=[
                [p - 1 for p in self.params.sigma[dm][1:]] for dm in self.params.DM
            ],
        )
