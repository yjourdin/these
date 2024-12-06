from collections.abc import Sequence

import numpy as np
from mcda.relations import PreferenceStructure
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, value

from ...constants import EPSILON
from ...models import ModelEnum, group_model
from ...performance_table.normal_performance_table import NormalPerformanceTable
from ...srmp.model import (
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
        gamma: float = EPSILON,
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

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        self.param["profiles_shared"] = SRMPParamEnum.PROFILES in self.shared_params
        self.param["weights_shared"] = SRMPParamEnum.WEIGHTS in self.shared_params
        # List of alternatives
        self.param["A_star"] = self.alternatives.alternatives
        # List of criteria
        self.param["M"] = self.alternatives.criteria
        # Number of profiles
        self.param["k"] = len(self.lexicographic_order)
        # List of DMs
        self.param["DM"] = range(len(self.preference_relations))
        # Indices of profiles
        self.param["profile_indices"] = list(range(1, self.param["k"] + 1))
        # Lexicographic order
        self.param["lexicographic_order"] = [0] + [
            profile + 1 for profile in self.lexicographic_order
        ]
        # Binary comparisons with preference
        self.param["preference_relations_indices"] = [
            range(len(self.preference_relations[dm])) for dm in self.param["DM"]
        ]
        # Binary comparisons with indifference
        self.param["indifference_relations_indices"] = [
            range(len(self.indifference_relations[dm])) for dm in self.param["DM"]
        ]

        #############
        # Variables #
        #############

        # Weights
        self.var["w"] = LpVariable.dicts(
            "Weight",
            (
                self.param["DM"],
                self.param["M"],
            ),
            lowBound=0,
            upBound=1,
        )
        # Reference profiles
        self.var["p"] = LpVariable.dicts(
            "Profile",
            (
                self.param["DM"],
                self.param["profile_indices"],
                self.param["M"],
            ),
        )
        # Local concordance to a reference point
        self.var["delta"] = LpVariable.dicts(
            "LocalConcordance",
            (
                self.param["DM"],
                self.param["A_star"],
                self.param["profile_indices"],
                self.param["M"],
            ),
            cat=LpBinary,
        )
        # Weighted local concordance to a reference point
        self.var["omega"] = LpVariable.dicts(
            "WeightedLocalConcordance",
            (
                self.param["DM"],
                self.param["A_star"],
                self.param["profile_indices"],
                self.param["M"],
            ),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        self.var["s"] = {}
        for dm in self.param["DM"]:
            self.var["s"][dm] = LpVariable.dicts(
                f"PreferenceRankingVariable_{dm}",
                (
                    self.param["preference_relations_indices"][dm],
                    [0] + self.param["profile_indices"],
                ),
                cat=LpBinary,
            )

        # Variables used to model the ranking rule with indifference relations
        if self.inconsistencies:
            self.var["s_star"] = {}
            for dm in self.param["DM"]:
                self.var["s_star"][dm] = LpVariable.dicts(
                    f"IndifferenceRankingVariable_{dm}",
                    self.param["indifference_relations_indices"][dm],
                    cat=LpBinary,
                )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            self.prob += lpSum(
                [
                    [
                        self.var["s"][dm][index][0]
                        for index in self.param["preference_relations_indices"][dm]
                    ]
                    for dm in self.param["DM"]
                ]
            ) + lpSum(
                [
                    [
                        self.var["s_star"][dm][index]
                        for index in self.param["indifference_relations_indices"][dm]
                    ]
                    for dm in self.param["DM"]
                ]
            )

        ###############
        # Constraints #
        ###############

        # Normalized weights

        for dm in self.param["DM"]:
            self.prob += lpSum([self.var["w"][dm][j] for j in self.param["M"]]) == 1

        for j in self.param["M"]:
            for dm in self.param["DM"]:
                # Non-zero weights
                self.prob += self.var["w"][dm][j] >= self.gamma

            for dm in self.param["DM"]:
                # Constraints on the reference profiles
                self.prob += self.var["p"][dm][1][j] >= 0
                self.prob += self.var["p"][dm][self.param["k"]][j] <= 1

            for h in self.param["profile_indices"]:
                if h != self.param["k"]:
                    for dm in self.param["DM"]:
                        # Dominance between the reference profiles
                        self.prob += (
                            self.var["p"][dm][h + 1][j] >= self.var["p"][dm][h][j]
                        )

                for a in self.param["A_star"]:
                    for dm in self.param["DM"]:
                        # Constraints on the local concordances
                        self.prob += (
                            self.alternatives.cell[a, j] - self.var["p"][dm][h][j]
                            >= self.var["delta"][dm][a][h][j] - 1
                        )
                        self.prob += (
                            self.var["delta"][dm][a][h][j]
                            >= self.alternatives.cell[a, j]
                            - self.var["p"][dm][h][j]
                            + self.gamma
                        )

                    for dm in self.param["DM"]:
                        # Constraints on the weighted local concordances
                        self.prob += (
                            self.var["omega"][dm][a][h][j] <= self.var["w"][dm][j]
                        )
                        self.prob += self.var["omega"][dm][a][h][j] >= 0
                        self.prob += (
                            self.var["omega"][dm][a][h][j]
                            <= self.var["delta"][dm][a][h][j]
                        )
                        self.prob += (
                            self.var["omega"][dm][a][h][j]
                            >= self.var["delta"][dm][a][h][j] + self.var["w"][dm][j] - 1
                        )

        # Constraints on the preference ranking variables
        for dm in self.param["DM"]:
            for index in self.param["preference_relations_indices"][dm]:
                if not self.inconsistencies:
                    self.prob += (
                        self.var["s"][dm][index][self.param["lexicographic_order"][0]]
                        == 1
                    )
                self.prob += (
                    self.var["s"][dm][index][
                        self.param["lexicographic_order"][self.param["k"]]
                    ]
                    == 0
                )

        for h in self.param["profile_indices"]:
            # Constraints on the preferences
            for dm in self.param["DM"]:
                for index, relation in enumerate(self.preference_relations[dm]):
                    a, b = relation.a, relation.b
                    self.prob += lpSum(
                        [
                            self.var["omega"][dm][a][
                                self.param["lexicographic_order"][h]
                            ][j]
                            for j in self.param["M"]
                        ]
                    ) >= (
                        lpSum(
                            [
                                self.var["omega"][dm][b][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        )
                        + self.gamma
                        - self.var["s"][dm][index][self.param["lexicographic_order"][h]]
                        * (1 + self.gamma)
                        - (
                            1
                            - self.var["s"][dm][index][
                                self.param["lexicographic_order"][h - 1]
                            ]
                        )
                    )

                    self.prob += lpSum(
                        [
                            self.var["omega"][dm][a][
                                self.param["lexicographic_order"][h]
                            ][j]
                            for j in self.param["M"]
                        ]
                    ) >= (
                        lpSum(
                            [
                                self.var["omega"][dm][b][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        )
                        - (
                            1
                            - self.var["s"][dm][index][
                                self.param["lexicographic_order"][h]
                            ]
                        )
                        - (
                            1
                            - self.var["s"][dm][index][
                                self.param["lexicographic_order"][h - 1]
                            ]
                        )
                    )

                    self.prob += lpSum(
                        [
                            self.var["omega"][dm][a][
                                self.param["lexicographic_order"][h]
                            ][j]
                            for j in self.param["M"]
                        ]
                    ) <= (
                        lpSum(
                            [
                                self.var["omega"][dm][b][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        )
                        + (
                            1
                            - self.var["s"][dm][index][
                                self.param["lexicographic_order"][h]
                            ]
                        )
                        + (
                            1
                            - self.var["s"][dm][index][
                                self.param["lexicographic_order"][h - 1]
                            ]
                        )
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(self.indifference_relations[dm]):
                    a, b = relation.a, relation.b
                    if not self.inconsistencies:
                        self.prob += lpSum(
                            [
                                self.var["omega"][dm][a][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        ) == lpSum(
                            [
                                self.var["omega"][dm][b][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        )
                    else:
                        self.prob += lpSum(
                            [
                                self.var["omega"][dm][a][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        ) <= (
                            lpSum(
                                [
                                    self.var["omega"][dm][b][
                                        self.param["lexicographic_order"][h]
                                    ][j]
                                    for j in self.param["M"]
                                ]
                            )
                            - (1 - self.var["s_star"][dm][index])
                        )

                        self.prob += lpSum(
                            [
                                self.var["omega"][dm][b][
                                    self.param["lexicographic_order"][h]
                                ][j]
                                for j in self.param["M"]
                            ]
                        ) <= (
                            lpSum(
                                [
                                    self.var["omega"][dm][a][
                                        self.param["lexicographic_order"][h]
                                    ][j]
                                    for j in self.param["M"]
                                ]
                            )
                            - (1 - self.var["s_star"][dm][index])
                        )

        # Constraint on shared parameters
        if self.param["profiles_shared"] or self.param["weights_shared"]:
            for j in self.param["M"]:
                for dm in self.param["DM"][:-1]:
                    if self.param["weights_shared"]:
                        self.prob += self.var["w"][dm][j] == self.var["w"][dm + 1][j]
                    if self.param["profiles_shared"]:
                        for h in self.param["profile_indices"]:
                            self.prob += (
                                self.var["p"][dm][h][j] == self.var["p"][dm + 1][h][j]
                            )

    def create_solution(self):
        weights = (
            np.array([value(self.var["w"][0][j]) for j in self.param["M"]])
            if self.param["weights_shared"]
            else [
                np.array([value(self.var["w"][dm][j]) for j in self.param["M"]])
                for dm in self.param["DM"]
            ]
        )
        profiles = (
            NormalPerformanceTable(
                [
                    [value(self.var["p"][0][h][j]) for j in self.param["M"]]
                    for h in self.param["profile_indices"]
                ]
            )
            if self.param["profiles_shared"]
            else [
                NormalPerformanceTable(
                    [
                        [value(self.var["p"][dm][h][j]) for j in self.param["M"]]
                        for h in self.param["profile_indices"]
                    ]
                )
                for dm in self.param["DM"]
            ]
        )

        return group_model(
            ModelEnum.SRMP, self.shared_params
        )(
            group_size=len(self.param["DM"]),
            profiles=profiles,  # type: ignore
            weights=weights,  # type: ignore
            lexicographic_order=[  # type: ignore
                p - 1 for p in self.param["lexicographic_order"][1:]
            ],
        )
