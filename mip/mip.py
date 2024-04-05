from collections.abc import Sequence

from mcda.core.interfaces import Learner
from mcda.core.relations import PreferenceStructure
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, getSolver, lpSum, value

from performance_table.normal_performance_table import NormalPerformanceTable
from srmp.model import SRMPModel

solver = getSolver("PULP_CBC_CMD", msg=False)


class MIP(Learner[SRMPModel | None]):
    def __init__(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: PreferenceStructure,
        indifference_relations: PreferenceStructure,
        lexicographic_order: Sequence[int],
        gamma: float,
        inconsistencies: bool,
    ):
        self.alternatives = alternatives
        self.preference_relations = preference_relations
        self.indifference_relations = indifference_relations
        self.lexicographic_order = lexicographic_order
        self.gamma = gamma
        self.inconsistencies = inconsistencies

    def _learn(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: PreferenceStructure,
        indifference_relations: PreferenceStructure,
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
        # Binary comparisons with preference
        preference_relations_indices = range(len(preference_relations))
        # Binary comparisons with indifference
        indifference_relations_indices = range(len(indifference_relations))

        #############
        # Variables #
        #############

        # Weights
        w = LpVariable.dicts("Weight", M, lowBound=0, upBound=1)
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
            (A_star, profile_indices, M),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        s = LpVariable.dicts(
            "PreferenceRankingVariable",
            (
                preference_relations_indices,
                [0] + profile_indices,
            ),
            cat=LpBinary,
        )

        if self.inconsistencies:
            # Variables used to model the ranking rule with indifference
            # relations
            s_star = LpVariable.dicts(
                "IndifferenceRankingVariable",
                indifference_relations_indices,
                cat=LpBinary,
            )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum(
                [s[index][0] for index in preference_relations_indices]
            ) + lpSum([s_star[index] for index in indifference_relations_indices])

        ###############
        # Constraints #
        ###############

        # Normalized weights
        self.prob += lpSum([w[j] for j in M]) == 1

        for j in M:
            # Non-zero weights
            self.prob += w[j] >= self.gamma

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
                    self.prob += omega[a][h][j] <= w[j]
                    self.prob += omega[a][h][j] >= 0
                    self.prob += omega[a][h][j] <= delta[a][h][j]
                    self.prob += omega[a][h][j] >= delta[a][h][j] + w[j] - 1

        # Constraints on the preference ranking variables
        for index in preference_relations_indices:
            if not self.inconsistencies:
                self.prob += s[index][lexicographic_order[0]] == 1
            self.prob += s[index][lexicographic_order[k]] == 0

        for h in profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(preference_relations):
                a, b = relation.a, relation.b

                self.prob += lpSum(
                    [omega[a][lexicographic_order[h]][j] for j in M]
                ) >= (
                    lpSum([omega[b][lexicographic_order[h]][j] for j in M])
                    + self.gamma
                    - s[index][lexicographic_order[h]] * (1 + self.gamma)
                    - (1 - s[index][lexicographic_order[h - 1]])
                )

                self.prob += lpSum(
                    [omega[a][lexicographic_order[h]][j] for j in M]
                ) >= (
                    lpSum([omega[b][lexicographic_order[h]][j] for j in M])
                    - (1 - s[index][lexicographic_order[h]])
                    - (1 - s[index][lexicographic_order[h - 1]])
                )

                self.prob += lpSum(
                    [omega[a][lexicographic_order[h]][j] for j in M]
                ) <= (
                    lpSum([omega[b][lexicographic_order[h]][j] for j in M])
                    + (1 - s[index][lexicographic_order[h]])
                    + (1 - s[index][lexicographic_order[h - 1]])
                )

            # Constraints on the indifferences
            for index, relation in enumerate(indifference_relations):
                a, b = relation.a, relation.b
                if not self.inconsistencies:
                    self.prob += lpSum(
                        [omega[a][lexicographic_order[h]][j] for j in M]
                    ) == lpSum([omega[b][lexicographic_order[h]][j] for j in M])
                else:
                    self.prob += lpSum(
                        [omega[a][lexicographic_order[h]][j] for j in M]
                    ) <= (
                        lpSum([omega[b][lexicographic_order[h]][j] for j in M])
                        - (1 - s_star[index])
                    )

                    self.prob += lpSum(
                        [omega[b][lexicographic_order[h]][j] for j in M]
                    ) <= (
                        lpSum([omega[a][lexicographic_order[h]][j] for j in M])
                        - (1 - s_star[index])
                    )

        # Solve problem
        status = self.prob.solve(solver)

        if status != 1:
            return None

        # Compute optimum solution
        weights = [value(w[j]) for j in M]
        profiles = NormalPerformanceTable(
            [[value(p[h][j]) for j in M] for h in profile_indices]
        )

        return SRMPModel(profiles, weights, [p - 1 for p in lexicographic_order[1:]])

    def learn(self):
        return self._learn(
            self.alternatives,
            self.preference_relations,
            self.indifference_relations,
            self.lexicographic_order,
        )
