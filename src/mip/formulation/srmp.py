# pyright: reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
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

# from pyomo.core.base.param import (  # pyright: ignore[reportMissingTypeStubs]  # pyright: ignore[reportMissingTypeStubs]
#     IndexedParam,
#     ScalarParam,
# )
# from pyomo.core.base.set import (  # pyright: ignore[reportMissingTypeStubs]  # pyright: ignore[reportMissingTypeStubs]
#     FiniteScalarRangeSet,
#     IndexedSet,
# )
# from pyomo.core.base.var import (  # pyright: ignore[reportMissingTypeStubs]  # pyright: ignore[reportMissingTypeStubs]
#     IndexedVar,
# )
# from pyomo.environ import (  # pyright: ignore[reportMissingTypeStubs]
#     AbstractModel,
#     Binary,
#     Constraint,
#     Expression,
#     NonNegativeIntegers,
#     Objective,
#     Param,
#     PositiveIntegers,
#     RangeSet,
#     Set,
#     UnitInterval,
#     Var,
#     maximize,
#     quicksum,
# )
from src.constants import EPSILON
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.srmp.model import SRMPModel

from ..mip import MIP, D, MIPParams, MIPVars

# class _MIPSRMP(AbstractModel):
#     alternatives: IndexedSet
#     criteria: IndexedSet
#     A: IndexedParam
#     preference_relations: IndexedSet
#     indifference_relations: IndexedSet
#     K: ScalarParam
#     profiles: FiniteScalarRangeSet
#     extended_profiles: IndexedSet
#     lexicographic_order: IndexedParam
#     sigma: dict[int, int]
#     gamma: ScalarParam
#     w: IndexedVar
#     p: IndexedVar
#     delta: IndexedVar
#     omega: IndexedVar
#     s: IndexedVar
#     s_star: IndexedVar


# def srmp_model(inconsistencies: bool, best_fitness: float | None = None):
#     M = _MIPSRMP()

#     M.alternatives = Set()
#     M.criteria = Set()
#     M.A = Param(M.alternatives, M.criteria)

#     M.preference_relations = Set()
#     M.indifference_relations = Set()

#     M.K = Param(domain=NonNegativeIntegers)
#     M.profiles = RangeSet(M.K)
#     M.extended_profiles = {0} | M.profiles
#     M.lexicographic_order = Param(M.profiles, domain=PositiveIntegers)
#     M.sigma = {0: 0} + dict(M.lexicographic_order.items())

#     M.gamma = Param()

#     M.w = Var(M.criteria, domain=UnitInterval)
#     M.p = Var(M.profiles, M.criteria, domain=UnitInterval)
#     M.delta = Var(M.alternatives, M.profiles, M.criteria, domain=Binary)
#     M.omega = Var(M.alternatives, M.profiles, M.criteria, domain=UnitInterval)
#     M.s = Var(M.preference_relations, M.extended_profiles, domain=Binary)

#     # Constraints

#     # Normalized weights
#     M.normalized_weights = Constraint(expr=quicksum(M.w) == 1)

#     # Dominance between the reference profiles
#     def dominance(M: MIPSRMP, j: int, h: int) -> Expression:
#         return M.p[h + 1, j] >= M.p[h, j]

#     M.dominance = Constraint(M.criteria, M.profiles - {M.K}, rule=dominance)

#     # Constraints on the local concordances
#     def local_1(M: MIPSRMP, j: int, h: int, a: Any) -> Expression:
#         return M.A[a, j] - M.p[h, j] >= M.delta[a, h, j] - 1

#     def local_2(M: MIPSRMP, j: int, h: int, a: Any) -> Expression:
#         return M.delta[a, h, j] >= M.A[a, j] - M.p[h, j] + M.gamma

#     M.local_1 = Constraint(M.criteria, M.profiles, M.alternatives, rule=local_1)
#     M.local_2 = Constraint(M.criteria, M.profiles, M.alternatives, rule=local_2)

#     # Constraints on the weighted local concordances
#     def weighted_local_1(M: MIPSRMP, j: int, h: int, a: Any):
#         return M.omega[a, h, j] <= M.w[j]

#     def weighted_local_2(M: MIPSRMP, j: int, h: int, a: Any):
#         return M.omega[a, h, j] <= M.delta[a, h, j]

#     def weighted_local_3(M: MIPSRMP, j: int, h: int, a: Any):
#         return M.omega[a, h, j] >= M.delta[a, h, j] + M.w[j] - 1

#     M.weighted_local_1 = Constraint(
#         M.criteria, M.profiles, M.alternatives, rule=weighted_local_1
#     )
#     M.weighted_local_2 = Constraint(
#         M.criteria, M.profiles, M.alternatives, rule=weighted_local_2
#     )
#     M.weighted_local_3 = Constraint(
#         M.criteria, M.profiles, M.alternatives, rule=weighted_local_3
#     )

#     # Constraints on the preference ranking variables
#     if not inconsistencies:

#         def preference_ranking_1(M: MIPSRMP, r: PreferenceRelation):
#             return M.s[r, 0] == 1

#         M.preference_ranking_1 = Constraint(
#             M.preference_relations, rule=preference_ranking_1
#         )

#     def preference_ranking_2(M: MIPSRMP, r: int):
#         return M.s[r, M.K] == 1

#     M.preference_ranking_2 = Constraint(
#         M.preference_relations, rule=preference_ranking_2
#     )

#     def omega_sum(M: MIPSRMP, a: Any, h: int):
#         return quicksum(M.omega[a, M.sigma[h], j] for j in M.criteria)

#     # Constraints on the preferences
#     def preference_1(M: MIPSRMP, h: int, r: PreferenceRelation):
#         return omega_sum(M, r.a, M.sigma[h]) >= omega_sum(
#             M, r.b, M.sigma[h]
#         ) + M.gamma - M.s[r, M.sigma[h]] * (1 + M.gamma) - (1 - M.s[r, M.sigma[h - 1]])

#     def preference_2(M: MIPSRMP, h: int, r: PreferenceRelation):
#         return omega_sum(M, r.a, M.sigma[h]) >= omega_sum(M, r.b, M.sigma[h]) - (
#             1 - M.s[r, M.sigma[h]]
#         ) - (1 - M.s[r, M.sigma[h - 1]])

#     def preference_3(M: MIPSRMP, h: int, r: PreferenceRelation):
#         return omega_sum(M, r.a, M.sigma[h]) >= omega_sum(M, r.b, M.sigma[h]) + (
#             1 - M.s[r, M.sigma[h]]
#         ) + (1 - M.s[r, M.sigma[h - 1]])

#     M.preference_1 = Constraint(M.profiles, M.preference_relations, rule=preference_1)
#     M.preference_2 = Constraint(M.profiles, M.preference_relations, rule=preference_2)
#     M.preference_3 = Constraint(M.profiles, M.preference_relations, rule=preference_3)

#     # Constraints on the indifferences
#     if not inconsistencies:

#         def indifference_0(M: MIPSRMP, h: int, r: IndifferenceRelation):
#             return omega_sum(M, r.a, M.sigma[h]) == omega_sum(M, r.b, M.sigma[h])

#         M.indifference_0 = Constraint(
#             M.profiles, M.indifference_relations, rule=indifference_0
#         )
#     else:
#         M.s_star = Var(M.indifference_relations, domain=Binary)

#         objective = quicksum(M.s[i, 0] for i in M.preference_relations) + quicksum(
#             M.s_star
#         )

#         M.OBJ = Objective(
#             expr=objective,
#             sense=maximize,
#         )

#         def indifference_1(M: MIPSRMP, h: int, r: IndifferenceRelation):
#             return omega_sum(M, r.a, M.sigma[h]) <= omega_sum(M, r.b, M.sigma[h]) - (
#                 1 - M.s_star[r]
#             )

#         def indifference_2(M: MIPSRMP, h: int, r: IndifferenceRelation):
#             return omega_sum(M, r.b, M.sigma[h]) <= omega_sum(M, r.a, M.sigma[h]) - (
#                 1 - M.s_star[r]
#             )

#         M.indifference_1 = Constraint(
#             M.profiles, M.indifference_relations, rule=indifference_1
#         )
#         M.indifference_2 = Constraint(
#             M.profiles, M.indifference_relations, rule=indifference_2
#         )

#         if best_fitness is not None:
#             M.better_fitness = objective >= best_fitness + M.gamma

#     return M


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


@dataclass(kw_only=True)
class MIPSRMP(MIP[SRMPModel, MIPSRMPVars, MIPSRMPParams]):
    alternatives: NormalPerformanceTable
    preference_relations: list[P]
    indifference_relations: list[I]
    lexicographic_order: Sequence[int]
    gamma: float = EPSILON
    inconsistencies: bool = True
    best_fitness: float | None = None

    def create_parameters(self):
        self.params = MIPSRMPParams(
            A=self.alternatives.alternatives,  # type: ignore
            M=self.alternatives.criteria,  # type: ignore
            lexicographic_order=self.lexicographic_order,
        )

    def create_variables(self):
        # Binary comparisons with preference
        preference_relations_indices = range(len(self.preference_relations))
        # Binary comparisons with indifference
        indifference_relations_indices = range(len(self.indifference_relations))

        self.vars = MIPSRMPVars(
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
                (preference_relations_indices, [0] + self.params.profile_indices),
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

    def create_problem(self):
        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum([s[0] for s in self.vars["s"].values()]) + lpSum([
                s_star for s_star in self.vars["s_star"].values()
            ])

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
                    self.prob += self.vars["omega"][a][h][j] >= 0
                    self.prob += (
                        self.vars["omega"][a][h][j] <= self.vars["delta"][a][h][j]
                    )
                    self.prob += (
                        self.vars["omega"][a][h][j]
                        >= self.vars["delta"][a][h][j] + self.vars["w"][j] - 1
                    )

        # Constraints on the preference ranking variables
        for s in self.vars["s"]:
            if not self.inconsistencies:
                self.prob += s[self.params.sigma[0]] == 1
            self.prob += s[self.params.sigma[self.params.k]] == 0

        for h in self.params.profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(self.preference_relations):
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
                    - self.vars["s"][index][self.params.sigma[h]] * (1 + self.gamma)
                    - (1 - self.vars["s"][index][self.params.sigma[h - 1]])
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.params.sigma[h]][j]
                    for j in self.params.M
                ]) >= (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    - (1 - self.vars["s"][index][self.params.sigma[h]])
                    - (1 - self.vars["s"][index][self.params.sigma[h - 1]])
                )

                self.prob += lpSum([
                    self.vars["omega"][a][self.params.sigma[h]][j]
                    for j in self.params.M
                ]) <= (
                    lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                    + (1 - self.vars["s"][index][self.params.sigma[h]])
                    + (1 - self.vars["s"][index][self.params.sigma[h - 1]])
                )

            # Constraints on the indifferences
            for index, relation in enumerate(self.indifference_relations):
                a, b = relation.a, relation.b
                if not self.inconsistencies:
                    self.prob += lpSum([
                        self.vars["omega"][a][self.params.sigma[h]][j]
                        for j in self.params.M
                    ]) == lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ])
                else:
                    self.prob += lpSum([
                        self.vars["omega"][a][self.params.sigma[h]][j]
                        for j in self.params.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][b][self.params.sigma[h]][j]
                            for j in self.params.M
                        ])
                        - (1 - self.vars["s_star"][index])
                    )

                    self.prob += lpSum([
                        self.vars["omega"][b][self.params.sigma[h]][j]
                        for j in self.params.M
                    ]) <= (
                        lpSum([
                            self.vars["omega"][a][self.params.sigma[h]][j]
                            for j in self.params.M
                        ])
                        - (1 - self.vars["s_star"][index])
                    )

        if self.inconsistencies and (self.best_fitness is not None):
            self.prob += (
                lpSum([s[0] for s in self.vars["s"].values()])
                + lpSum([s_star for s_star in self.vars["s_star"].values()])
                >= self.best_fitness + self.gamma
            )

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
