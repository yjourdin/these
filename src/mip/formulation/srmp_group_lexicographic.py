from collections.abc import Sequence

from mcda.relations import PreferenceStructure
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, value

from ...performance_table.normal_performance_table import NormalPerformanceTable
from ...srmp.model import (
    SRMP_model,
    SRMPGroupModelLexicographic,
    SRMPGroupModelProfilesLexicographic,
    SRMPGroupModelWeightsLexicographic,
    SRMPGroupModelWeightsProfilesLexicographic,
    SRMPParamEnum,
)
from ..mip import MIP


class MIPSRMPGroupLexicographicOrder(
    MIP[
        SRMPGroupModelWeightsProfilesLexicographic
        | SRMPGroupModelWeightsLexicographic
        | SRMPGroupModelProfilesLexicographic
        | SRMPGroupModelLexicographic
    ]
):
    def __init__(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: list[PreferenceStructure],
        indifference_relations: list[PreferenceStructure],
        lexicographic_order: Sequence[int],
        shared_params: set[SRMPParamEnum] = set(),
        gamma: float = 0.001,
        inconsistencies: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alternatives = alternatives
        self.preference_relations = preference_relations
        self.indifference_relations = indifference_relations
        self.lexicographic_order = lexicographic_order
        self.shared_params = shared_params
        self.inconsistencies = inconsistencies
        self.gamma = gamma

    def _learn(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: list[PreferenceStructure],
        indifference_relations: list[PreferenceStructure],
        lexicographic_order: Sequence[int],
        shared_params: set[SRMPParamEnum] = set(),
    ):
        profiles_shared = SRMPParamEnum.PROFILES in shared_params
        weights_shared = SRMPParamEnum.WEIGHTS in shared_params

        ##############
        # Parameters #
        ##############

        # List of alternatives
        A_star = alternatives.alternatives
        # List of criteria
        M = alternatives.criteria
        # Number of profiles
        k = len(lexicographic_order)
        # List of DMs
        DM = range(len(preference_relations))
        DM_profiles = DM if not profiles_shared else range(1)
        DM_weights = DM if not weights_shared else range(1)
        DM_profiles_weights = (
            DM if (not profiles_shared) and (not weights_shared) else range(1)
        )
        # Indices of profiles
        profile_indices = list(range(1, k + 1))
        # Lexicographic order
        lexicographic_order = [0] + [profile + 1 for profile in lexicographic_order]
        # Binary comparisons with preference
        preference_relations_indices = [
            range(len(preference_relations[dm])) for dm in DM
        ]
        # Binary comparisons with indifference
        indifference_relations_indices = [
            range(len(indifference_relations[dm])) for dm in DM
        ]

        def dm_profiles(i):
            return i if not profiles_shared else 0

        def dm_weights(i):
            return i if not weights_shared else 0

        def dm_profiles_weights(i):
            return i if (not profiles_shared) and (not weights_shared) else 0

        #############
        # Variables #
        #############

        # Weights
        w = LpVariable.dicts(
            "Weight",
            (
                DM_weights,
                M,
            ),
            lowBound=0,
            upBound=1,
        )
        # Reference profiles
        p = LpVariable.dicts(
            "Profile",
            (
                DM_profiles,
                profile_indices,
                M,
            ),
        )
        # Local concordance to a reference point
        delta = LpVariable.dicts(
            "LocalConcordance",
            (
                DM_profiles,
                A_star,
                profile_indices,
                M,
            ),
            cat=LpBinary,
        )
        # Weighted local concordance to a reference point
        omega = LpVariable.dicts(
            "WeightedLocalConcordance",
            (
                DM_profiles_weights,
                A_star,
                profile_indices,
                M,
            ),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        s = {}
        for dm in DM:
            s[dm] = LpVariable.dicts(
                "PreferenceRankingVariable",
                (
                    preference_relations_indices[dm],
                    [0] + profile_indices,
                ),
                cat=LpBinary,
            )

        # Variables used to model the ranking rule with indifference relations
        if self.inconsistencies:
            s_star = {}
            for dm in DM:
                s_star[dm] = LpVariable.dicts(
                    "IndifferenceRankingVariable",
                    indifference_relations_indices[dm],
                    cat=LpBinary,
                )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum(
                [
                    [s[dm][index][0] for index in preference_relations_indices]
                    for dm in DM
                ]
            ) + lpSum(
                [
                    [s_star[dm][index] for index in indifference_relations_indices]
                    for dm in DM
                ]
            )

        ###############
        # Constraints #
        ###############

        # Normalized weights

        for dm in DM_weights:
            self.prob += lpSum([w[dm][j] for j in M]) == 1

        for j in M:
            for dm in DM_weights:
                # Non-zero weights
                self.prob += w[dm][j] >= self.gamma

            for dm in DM_profiles:
                # Constraints on the reference profiles
                self.prob += p[dm][1][j] >= 0
                self.prob += p[dm][k][j] <= 1

            for h in profile_indices:
                if h != k:
                    for dm in DM_profiles:
                        # Dominance between the reference profiles
                        self.prob += p[dm][h + 1][j] >= p[dm][h][j]

                for a in A_star:
                    for dm in DM_profiles:
                        # Constraints on the local concordances
                        self.prob += (
                            alternatives.cell[a, j] - p[dm][h][j]
                            >= delta[dm][a][h][j] - 1
                        )
                        self.prob += (
                            delta[dm][a][h][j]
                            >= alternatives.cell[a, j] - p[dm][h][j] + self.gamma
                        )

                    for dm in DM_profiles_weights:
                        # Constraints on the weighted local concordances
                        self.prob += omega[dm][a][h][j] <= w[dm_weights(dm)][j]
                        self.prob += omega[dm][a][h][j] >= 0
                        self.prob += (
                            omega[dm][a][h][j] <= delta[dm_profiles(dm)][a][h][j]
                        )
                        self.prob += (
                            omega[dm][a][h][j]
                            >= delta[dm_profiles(dm)][a][h][j] + w[j] - 1
                        )

        # Constraints on the preference ranking variables
        for dm in DM:
            for index in preference_relations_indices[dm]:
                if not self.inconsistencies:
                    self.prob += s[dm][index][lexicographic_order[0]] == 1
                self.prob += s[dm][index][lexicographic_order[k]] == 0

        for h in profile_indices:
            # Constraints on the preferences
            for dm in DM:
                for index, relation in enumerate(preference_relations[dm]):
                    a, b = relation.a, relation.b

                    self.prob += lpSum(
                        [
                            omega[dm_profiles_weights(dm)][a][lexicographic_order[h]][j]
                            for j in M
                        ]
                    ) >= (
                        lpSum(
                            [
                                omega[dm_profiles_weights(dm)][b][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        )
                        + self.gamma
                        - s[dm][index][lexicographic_order[h]] * (1 + self.gamma)
                        - (1 - s[dm][index][lexicographic_order[h - 1]])
                    )

                    self.prob += lpSum(
                        [
                            omega[dm_profiles_weights(dm)][a][lexicographic_order[h]][j]
                            for j in M
                        ]
                    ) >= (
                        lpSum(
                            [
                                omega[dm_profiles_weights(dm)][b][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        )
                        - (1 - s[dm][index][lexicographic_order[h]])
                        - (1 - s[dm][index][lexicographic_order[h - 1]])
                    )

                    self.prob += lpSum(
                        [
                            omega[dm_profiles_weights(dm)][a][lexicographic_order[h]][j]
                            for j in M
                        ]
                    ) <= (
                        lpSum(
                            [
                                omega[dm_profiles_weights(dm)][b][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        )
                        + (1 - s[dm][index][lexicographic_order[h]])
                        + (1 - s[dm][index][lexicographic_order[h - 1]])
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(indifference_relations[dm]):
                    a, b = relation.a, relation.b
                    if not self.inconsistencies:
                        self.prob += lpSum(
                            [
                                omega[dm_profiles_weights(dm)][a][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        ) == lpSum(
                            [
                                omega[dm_profiles_weights(dm)][b][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        )
                    else:
                        self.prob += lpSum(
                            [
                                omega[dm_profiles_weights(dm)][a][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        ) <= (
                            lpSum(
                                [
                                    omega[dm_profiles_weights(dm)][b][
                                        lexicographic_order[h]
                                    ][j]
                                    for j in M
                                ]
                            )
                            - (1 - s_star[dm][index])
                        )

                        self.prob += lpSum(
                            [
                                omega[dm_profiles_weights(dm)][b][
                                    lexicographic_order[h]
                                ][j]
                                for j in M
                            ]
                        ) <= (
                            lpSum(
                                [
                                    omega[dm_profiles_weights(dm)][a][
                                        lexicographic_order[h]
                                    ][j]
                                    for j in M
                                ]
                            )
                            - (1 - s_star[dm][index])
                        )

        # Solve problem
        status = self.prob.solve(self.solver)

        if status != 1:
            return None

        # Compute optimum solution
        weights = (
            [value(w[0][j]) for j in M]
            if weights_shared
            else [[value(w[dm][j]) for j in M] for dm in DM]
        )
        profiles = (
            NormalPerformanceTable(
                [[value(p[0][h][j]) for j in M] for h in profile_indices]
            )
            if profiles_shared
            else [
                NormalPerformanceTable(
                    [[value(p[dm][h][j]) for j in M] for h in profile_indices]
                )
                for dm in DM
            ]
        )

        return SRMP_model(shared_params)(
            size=len(DM),
            profiles=profiles,  # type: ignore
            weights=weights,  # type: ignore
            lexicographic_order=[
                p - 1 for p in lexicographic_order[1:]
            ],  # type: ignore
        )

    def learn(self):
        return self._learn(
            self.alternatives,
            self.preference_relations,
            self.indifference_relations,
            self.lexicographic_order,
        )
