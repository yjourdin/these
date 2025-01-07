from collections.abc import Sequence

import numpy as np
from mcda.relations import PreferenceStructure
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, value

from ...constants import EPSILON
from ...performance_table.normal_performance_table import NormalPerformanceTable
from ...srmp.model import SRMPModel
from ..mip import MIP


class MIPSRMP(MIP[SRMPModel]):
    def __init__(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: PreferenceStructure,
        indifference_relations: PreferenceStructure,
        lexicographic_order: Sequence[int],
        gamma: float = EPSILON,
        inconsistencies: bool = True,
        best_fitness: float | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alternatives = alternatives
        self.preference_relations = preference_relations
        self.indifference_relations = indifference_relations
        self.lexicographic_order = lexicographic_order
        self.inconsistencies = inconsistencies
        self.gamma = gamma
        self.best_fitness = best_fitness

    def create_problem(self):
        ##############
        # Parameters #
        ##############

        # List of alternatives
        self.param["A"] = self.alternatives.alternatives
        # List of criteria
        self.param["M"] = self.alternatives.criteria
        # Number of profiles
        self.param["k"] = len(self.lexicographic_order)
        # Indices of profiles
        self.param["profile_indices"] = list(range(1, self.param["k"] + 1))
        # Lexicographic order
        self.param["sigma"] = [0] + [
            profile + 1 for profile in self.lexicographic_order
        ]
        # Binary comparisons with preference
        preference_relations_indices = range(len(self.preference_relations))
        # Binary comparisons with indifference
        indifference_relations_indices = range(len(self.indifference_relations))

        #############
        # Variables #
        #############

        # Weights
        self.var["w"] = LpVariable.dicts(
            "Weight", self.param["M"], lowBound=0, upBound=1
        )
        # Reference profiles
        self.var["p"] = LpVariable.dicts(
            "Profile",
            (self.param["profile_indices"], self.param["M"]),
            lowBound=0,
            upBound=1,
        )
        # Local concordance to a reference point
        self.var["delta"] = LpVariable.dicts(
            "LocalConcordance",
            (self.param["A"], self.param["profile_indices"], self.param["M"]),
            cat=LpBinary,
        )
        # Weighted local concordance to a reference point
        self.var["omega"] = LpVariable.dicts(
            "WeightedLocalConcordance",
            (self.param["A"], self.param["profile_indices"], self.param["M"]),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        self.var["s"] = LpVariable.dicts(
            "PreferenceRankingVariable",
            (
                preference_relations_indices,
                [0] + self.param["profile_indices"],
            ),
            cat=LpBinary,
        )

        if self.inconsistencies:
            # Variables used to model the ranking rule with indifference
            # relations
            self.var["s_star"] = LpVariable.dicts(
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
                [self.var["s"][index][0] for index in preference_relations_indices]
            ) + lpSum(
                [self.var["s_star"][index] for index in indifference_relations_indices]
            )

        ###############
        # Constraints #
        ###############

        # Normalized weights
        self.prob += lpSum([self.var["w"][j] for j in self.param["M"]]) == 1

        for j in self.param["M"]:
            # Non-zero weights
            # self.prob += self.var["w"][j] >= self.gamma

            # Constraints on the reference profiles
            # self.prob += self.var["p"][1][j] >= 0
            # self.prob += self.var["p"][self.param["k"]][j] <= 1

            for h in self.param["profile_indices"]:
                if h != self.param["k"]:
                    # Dominance between the reference profiles
                    self.prob += self.var["p"][h + 1][j] >= self.var["p"][h][j]

                for a in self.param["A"]:
                    # Constraints on the local concordances
                    self.prob += (
                        self.alternatives.cell[a, j] - self.var["p"][h][j]
                        >= self.var["delta"][a][h][j] - 1
                    )
                    self.prob += (
                        self.var["delta"][a][h][j]
                        >= self.alternatives.cell[a, j]
                        - self.var["p"][h][j]
                        + self.gamma
                    )

                    # Constraints on the weighted local concordances
                    self.prob += self.var["omega"][a][h][j] <= self.var["w"][j]
                    self.prob += self.var["omega"][a][h][j] >= 0
                    self.prob += (
                        self.var["omega"][a][h][j] <= self.var["delta"][a][h][j]
                    )
                    self.prob += (
                        self.var["omega"][a][h][j]
                        >= self.var["delta"][a][h][j] + self.var["w"][j] - 1
                    )

        # Constraints on the preference ranking variables
        for index in preference_relations_indices:
            if not self.inconsistencies:
                self.prob += self.var["s"][index][self.param["sigma"][0]] == 1
            self.prob += self.var["s"][index][self.param["sigma"][self.param["k"]]] == 0

        for h in self.param["profile_indices"]:
            # Constraints on the preferences
            for index, relation in enumerate(self.preference_relations):
                a, b = relation.a, relation.b

                self.prob += lpSum(
                    [
                        self.var["omega"][a][self.param["sigma"][h]][j]
                        for j in self.param["M"]
                    ]
                ) >= (
                    lpSum(
                        [
                            self.var["omega"][b][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    )
                    + self.gamma
                    - self.var["s"][index][self.param["sigma"][h]] * (1 + self.gamma)
                    - (1 - self.var["s"][index][self.param["sigma"][h - 1]])
                )

                self.prob += lpSum(
                    [
                        self.var["omega"][a][self.param["sigma"][h]][j]
                        for j in self.param["M"]
                    ]
                ) >= (
                    lpSum(
                        [
                            self.var["omega"][b][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    )
                    - (1 - self.var["s"][index][self.param["sigma"][h]])
                    - (1 - self.var["s"][index][self.param["sigma"][h - 1]])
                )

                self.prob += lpSum(
                    [
                        self.var["omega"][a][self.param["sigma"][h]][j]
                        for j in self.param["M"]
                    ]
                ) <= (
                    lpSum(
                        [
                            self.var["omega"][b][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    )
                    + (1 - self.var["s"][index][self.param["sigma"][h]])
                    + (1 - self.var["s"][index][self.param["sigma"][h - 1]])
                )

            # Constraints on the indifferences
            for index, relation in enumerate(self.indifference_relations):
                a, b = relation.a, relation.b
                if not self.inconsistencies:
                    self.prob += lpSum(
                        [
                            self.var["omega"][a][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    ) == lpSum(
                        [
                            self.var["omega"][b][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    )
                else:
                    self.prob += lpSum(
                        [
                            self.var["omega"][a][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    ) <= (
                        lpSum(
                            [
                                self.var["omega"][b][self.param["sigma"][h]][j]
                                for j in self.param["M"]
                            ]
                        )
                        - (1 - self.var["s_star"][index])
                    )

                    self.prob += lpSum(
                        [
                            self.var["omega"][b][self.param["sigma"][h]][j]
                            for j in self.param["M"]
                        ]
                    ) <= (
                        lpSum(
                            [
                                self.var["omega"][a][self.param["sigma"][h]][j]
                                for j in self.param["M"]
                            ]
                        )
                        - (1 - self.var["s_star"][index])
                    )

        if self.inconsistencies and (self.best_fitness is not None):
            self.prob += lpSum(
                [self.var["s"][index][0] for index in preference_relations_indices]
            ) + lpSum(
                [self.var["s_star"][index] for index in indifference_relations_indices]
            ) >= self.best_fitness + self.gamma

    def create_solution(self):
        weights = np.array([value(self.var["w"][j]) for j in self.param["M"]])
        profiles = NormalPerformanceTable(
            [
                [value(self.var["p"][h][j]) for j in self.param["M"]]
                for h in self.param["profile_indices"]
            ]
        )

        return SRMPModel(
            profiles=profiles,
            weights=weights,
            lexicographic_order=[p - 1 for p in self.param["sigma"][1:]],
        )
