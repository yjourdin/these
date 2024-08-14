from collections.abc import Sequence

from mcda.internal.core.interfaces import Learner
from mcda.relations import PreferenceStructure
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpVariable,
    getSolver,
    listSolvers,
    lpSum,
    value,
)

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..srmp.model import SRMPModel


class MIP(Learner[list[SRMPModel] | None]):
    def __init__(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: list[PreferenceStructure],
        indifference_relations: list[PreferenceStructure],
        lexicographic_order: Sequence[int],
        gamma: float,
        inconsistencies: bool,
        seed: int,
        verbose: bool,
    ):
        self.alternatives = alternatives
        self.preference_relations = preference_relations
        self.indifference_relations = indifference_relations
        self.lexicographic_order = lexicographic_order
        self.gamma = gamma
        self.inconsistencies = inconsistencies

        if "GUROBI" in listSolvers(True):
            self.solver = getSolver(
                "GUROBI", msg=verbose, seed=seed % 2_000_000_000, threads=1
            )
        else:
            self.solver = getSolver("PULP_CBC_CMD", msg=verbose)

    def _learn(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: list[PreferenceStructure],
        indifference_relations: list[PreferenceStructure],
        lexicographic_order: Sequence[int],
    ):
        ##############
        # Parameters #
        ##############

        # List of alternatives
        A_star = alternatives.alternatives
        # List of criteria
        M = alternatives.criteria
        # Number of profiles
        k = len(lexicographic_order)
        # Indices of profiles
        profile_indices = list(range(1, k + 1))
        # Lexicographic order
        lexicographic_order = [0] + [profile + 1 for profile in lexicographic_order]
        # Number of DMs
        L = list(range(len(preference_relations)))
        # Binary comparisons with preference
        preference_relations_indices = [
            list(range(len(preference_relations[i]))) for i in L
        ]
        # Binary comparisons with indifference
        indifference_relations_indices = [
            list(range(len(indifference_relations[i]))) for i in L
        ]

        #############
        # Variables #
        #############

        # Weights
        w = LpVariable.dicts("Weight", (M, L), lowBound=0, upBound=1)
        # Reference profiles
        p = LpVariable.dicts("Profile", (profile_indices, M))
        # Local concordance to a reference point
        delta = LpVariable.dicts(
            "LocalConcordance",
            (A_star, profile_indices, M),
            cat=LpBinary,
        )
        # Weighted local concordance to a reference point
        omega = LpVariable.dicts(
            "WeightedLocalConcordance",
            (A_star, profile_indices, M, L),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        s = {}
        for i in L:
            s[i] = LpVariable.dicts(
                "PreferenceRankingVariable",
                (preference_relations_indices[i], [0] + profile_indices),
                cat=LpBinary,
            )

        if self.inconsistencies:
            # Variables used to model the ranking rule with indifference
            # relations
            s_star = {}
            for i in L:
                s_star[i] = LpVariable.dicts(
                    "IndifferenceRankingVariable",
                    indifference_relations_indices[i],
                    cat=LpBinary,
                )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum(
                [s[i][index][0] for index in preference_relations_indices[i] for i in L]
            ) + lpSum(
                [
                    s_star[i][index]
                    for index in indifference_relations_indices[i]
                    for i in L
                ]
            )

        ###############
        # Constraints #
        ###############

        # Normalized weights
        for i in L:
            self.prob += lpSum([w[j][i] for j in M]) == 1

        for j in M:
            # Non-zero weights
            for i in L:
                self.prob += w[j][i] >= self.gamma

            # Constraints on the reference profiles
            self.prob += p[1][j] >= 0
            self.prob += p[k][j] <= 1

            for h in profile_indices:
                if h != k:
                    # Dominance between the reference profiles
                    self.prob += p[h + 1][j] >= p[h][j]

                for a in A_star:
                    # Constraints on the local concordances
                    self.prob += alternatives.cell[a, j] - p[h][j] >= delta[a][h][j] - 1
                    self.prob += (
                        delta[a][h][j] >= alternatives.cell[a, j] - p[h][j] + self.gamma
                    )

                    # Constraints on the weighted local concordances
                    for i in L:
                        self.prob += omega[a][h][j][i] <= w[j][i]
                        self.prob += omega[a][h][j][i] >= 0
                        self.prob += omega[a][h][j][i] <= delta[a][h][j]
                        self.prob += omega[a][h][j][i] >= delta[a][h][j] + w[j][i] - 1

        # Constraints on the preference ranking variables
        for i in L:
            for index in preference_relations_indices[i]:
                if not self.inconsistencies:
                    self.prob += s[i][index][lexicographic_order[0]] == 1
                self.prob += s[i][index][lexicographic_order[k]] == 0

        for h in profile_indices:
            # Constraints on the preferences
            for i in L:
                for index, relation in enumerate(preference_relations[i]):
                    a, b = relation.a, relation.b

                    self.prob += lpSum(
                        [omega[a][lexicographic_order[h]][j][i] for j in M]
                    ) >= (
                        lpSum([omega[b][lexicographic_order[h]][j][i] for j in M])
                        + self.gamma
                        - s[i][index][lexicographic_order[h]] * (1 + self.gamma)
                        - (1 - s[i][index][lexicographic_order[h - 1]])
                    )

                    self.prob += lpSum(
                        [omega[a][lexicographic_order[h]][j][i] for j in M]
                    ) >= (
                        lpSum([omega[b][lexicographic_order[h]][j][i] for j in M])
                        - (1 - s[i][index][lexicographic_order[h]])
                        - (1 - s[i][index][lexicographic_order[h - 1]])
                    )

                    self.prob += lpSum(
                        [omega[a][lexicographic_order[h]][j][i] for j in M]
                    ) <= (
                        lpSum([omega[b][lexicographic_order[h]][j][i] for j in M])
                        + (1 - s[i][index][lexicographic_order[h]])
                        + (1 - s[i][index][lexicographic_order[h - 1]])
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(indifference_relations[i]):
                    a, b = relation.a, relation.b
                    if not self.inconsistencies:
                        self.prob += lpSum(
                            [omega[a][lexicographic_order[h]][j][i] for j in M]
                        ) == lpSum([omega[b][lexicographic_order[h]][j][i] for j in M])
                    else:
                        self.prob += lpSum(
                            [omega[a][lexicographic_order[h]][j][i] for j in M]
                        ) <= (
                            lpSum([omega[b][lexicographic_order[h]][j][i] for j in M])
                            - (1 - s_star[i][index])
                        )

                        self.prob += lpSum(
                            [omega[b][lexicographic_order[h]][j][i] for j in M]
                        ) <= (
                            lpSum([omega[a][lexicographic_order[h]][j][i] for j in M])
                            - (1 - s_star[i][index])
                        )

        # Solve problem
        status = self.prob.solve(self.solver)

        if status != 1:
            return None

        # Compute optimum solution
        weights = [[value(w[j][i]) for j in M] for i in L]
        profiles = NormalPerformanceTable(
            [[value(p[h][j]) for j in M] for h in profile_indices]
        )
        return [
            SRMPModel(profiles, weights[i], [p - 1 for p in lexicographic_order[1:]])
            for i in L
        ]

    def learn(self):
        return self._learn(
            self.alternatives,
            self.preference_relations,
            self.indifference_relations,
            self.lexicographic_order,
        )
