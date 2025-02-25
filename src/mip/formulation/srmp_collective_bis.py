import itertools
from collections.abc import Sequence
from typing import Any

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
from ..mip import MIP


class MIPSRMPCollective(MIP[SRMPModel]):
    def __init__(
        self,
        alternatives: NormalPerformanceTable,
        preference_relations: list[PreferenceStructure],
        indifference_relations: list[PreferenceStructure],
        lexicographic_order: Sequence[Sequence[int]],
        preferences_changed: list[int],
        preference_refused: list[PreferenceStructure],
        indifference_refused: list[PreferenceStructure],
        count_refused: list[int],
        gamma: float = EPSILON,
        best_objective: float | None = None,
        penalty: float = 0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.alternatives = alternatives
        self.preference_relations = preference_relations
        self.indifference_relations = indifference_relations
        self.lexicographic_order = lexicographic_order
        self.preferences_changed = preferences_changed
        self.preference_refused = preference_refused
        self.indifference_refused = indifference_refused
        self.count_refused = count_refused
        self.gamma = gamma
        self.best_objective = best_objective
        self.penalty = penalty

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
        # List of DMs
        self.param["DM"] = range(len(self.preference_relations))
        # List of models
        self.param["Models"] = range(len(self.preference_relations) + 1)
        # Index of collective model
        self.param["c"] = len(self.preference_relations)
        # Indices of profiles
        self.param["profile_indices"] = list(range(1, self.param["k"] + 1))
        # Lexicographic order
        self.param["sigma"] = [
            [0] + [profile + 1 for profile in self.lexicographic_order[model]]
            for model in self.param["Models"]
        ]
        # Binary comparisons with preference
        preference_relations_union = list(
            set(
                itertools.chain.from_iterable(
                    self.preference_relations[dm].relations for dm in self.param["DM"]
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
                    self.indifference_relations[dm].relations for dm in self.param["DM"]
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

        # Weights
        self.var["w"] = LpVariable.dicts(
            "Weight",
            (self.param["DM"], self.param["Models"]),
            lowBound=0,
            upBound=1,
        )
        # Reference profiles
        self.var["p"] = LpVariable.dicts(
            "Profile",
            (self.param["Models"], self.param["profile_indices"], self.param["M"]),
            lowBound=0,
            upBound=1,
        )
        # Local concordance to a reference point
        self.var["delta"] = LpVariable.dicts(
            "LocalConcordance",
            (
                self.param["Models"],
                self.param["A"],
                self.param["profile_indices"],
                self.param["M"],
            ),
            cat=LpBinary,
        )
        # Weighted local concordance to a reference point
        self.var["omega"] = LpVariable.dicts(
            "WeightedLocalConcordance",
            (
                self.param["Models"],
                self.param["A"],
                self.param["profile_indices"],
                self.param["M"],
            ),
            lowBound=0,
            upBound=1,
        )
        # Variables used to model the ranking rule with preference relations
        self.var["s"] = LpVariable.dicts(
            "PreferenceRankingVariable",
            (
                self.param["Models"],
                preference_relations_union_indices,
                [0] + self.param["profile_indices"],
            ),
            cat=LpBinary,
        )

        # Variables used to model the ranking rule with indifference relations
        self.var["s_star"] = LpVariable.dicts(
            "IndifferenceRankingVariable",
            (
                indifference_relations_union_indices,
                self.param["Models"],
            ),
            cat=LpBinary,
        )

        # Variables used to model the minimum number of preferences changes to get every DM consistent
        self.var["S"] = LpVariable(
            "MinimumPreferencesChanges",
            cat=LpInteger,
        )

        # Variables used to model inconsistencies
        self.var["R"] = LpVariable.dicts(
            "Inconsistencies",
            range(len(self.preference_refused)),
            cat=LpBinary,
        )

        # Variables used to model distance
        self.var["W_abs"] = LpVariable.dicts(
            "WeightAbsoluteDifference", self.param["DM"], self.param["M"]
        )
        self.var["W"] = LpVariable(
            "WeightDistance",
        )
        self.var["P_abs"] = LpVariable.dicts(
            "ProfileAbsoluteDifference",
            self.param["DM"],
            self.param["profile_indices"],
            self.param["M"],
        )
        self.var["P"] = LpVariable(
            "ProfileDistance",
        )

        ##############
        # LP problem #
        ##############

        self.prob = LpProblem("SRMP_Elicitation", LpMinimize)

        self.prob += (
            self.var["W"]
            + self.var["P"]
            + self.penalty
            + 2
            * (self.param["k"] + 1)
            * len(self.param["M"])  # type: ignore
            * (
                self.var["S"]
                + 2
                * (
                    len(preference_relations_union_indices)
                    + len(indifference_relations_union_indices)
                )
                * lpSum(self.var["R"])
            )
        )

        ###############
        # Constraints #
        ###############

        # Normalized weights
        for model in self.param["Models"]:
            self.prob += lpSum([self.var["w"][model][j] for j in self.param["M"]]) == 1

            for j in self.param["M"]:
                # Non-zero weights
                # self.prob += self.var["w"][j] >= self.gamma

                # Constraints on the reference profiles
                # self.prob += self.var["p"][1][j] >= 0
                # self.prob += self.var["p"][self.param["k"]][j] <= 1

                for h in self.param["profile_indices"]:
                    if h != self.param["k"]:
                        # Dominance between the reference profiles
                        self.prob += (
                            self.var["p"][model][h + 1][j] >= self.var["p"][model][h][j]
                        )

                    for a in self.param["A"]:
                        # Constraints on the local concordances
                        self.prob += (
                            self.alternatives.cell[a, j] - self.var["p"][model][h][j]
                            >= self.var["delta"][model][a][h][j] - 1
                        )
                        self.prob += (
                            self.var["delta"][model][a][h][j]
                            >= self.alternatives.cell[a, j]
                            - self.var["p"][model][h][j]
                            + self.gamma
                        )

                        # Constraints on the weighted local concordances
                        self.prob += (
                            self.var["omega"][model][a][h][j] <= self.var["w"][model][j]
                        )
                        self.prob += self.var["omega"][model][a][h][j] >= 0
                        self.prob += (
                            self.var["omega"][model][a][h][j]
                            <= self.var["delta"][model][a][h][j]
                        )
                        self.prob += (
                            self.var["omega"][model][a][h][j]
                            >= self.var["delta"][model][a][h][j]
                            + self.var["w"][model][j]
                            - 1
                        )

            # Constraints on the preference ranking variables
            for index in preference_relations_union_indices:
                self.prob += (
                    self.var["s"][model][index][self.param["sigma"][model][0]] == 0
                )

                # for h in self.param["profile_indices"]:
                #     self.prob += (
                #         self.var["s"][index][self.param["sigma"][h]]
                #         >= self.var["s"][index][self.param["sigma"][h - 1]]
                #     )

            for h in self.param["profile_indices"]:
                # Constraints on the preferences
                for index, relation in enumerate(preference_relations_union):
                    a, b = relation.a, relation.b
                    self.prob += lpSum([
                        self.var["omega"][model][a][self.param["sigma"][model][h]][j]
                        for j in self.param["M"]
                    ]) >= (
                        lpSum([
                            self.var["omega"][model][b][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ])
                        + self.gamma
                        - (1 + self.gamma)
                        * (
                            1
                            - self.var["s"][model][index][self.param["sigma"][model][h]]
                            + self.var["s"][model][index][
                                self.param["sigma"][model][h - 1]
                            ]
                        )
                    )

                    self.prob += lpSum([
                        self.var["omega"][model][a][self.param["sigma"][model][h]][j]
                        for j in self.param["M"]
                    ]) >= (
                        lpSum([
                            self.var["omega"][model][b][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ])
                        - self.var["s"][model][index][self.param["sigma"][model][h]]
                        - self.var["s"][model][index][self.param["sigma"][model][h - 1]]
                    )

                    self.prob += lpSum([
                        self.var["omega"][model][a][self.param["sigma"][model][h]][j]
                        for j in self.param["M"]
                    ]) <= (
                        lpSum([
                            self.var["omega"][model][b][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ])
                        + self.var["s"][model][index][self.param["sigma"][model][h]]
                        + self.var["s"][model][index][self.param["sigma"][model][h - 1]]
                    )

                # Constraints on the indifferences
                for index, relation in enumerate(indifference_relations_union):
                    a, b = relation.a, relation.b
                    if model == self.param["c"]:
                        self.prob += lpSum([
                            self.var["omega"][model][a][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ]) >= (
                            lpSum([
                                self.var["omega"][model][b][
                                    self.param["sigma"][model][h]
                                ][j]
                                for j in self.param["M"]
                            ])
                            - self.var["s_star"][model][index]
                        )

                        self.prob += lpSum([
                            self.var["omega"][model][a][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ]) <= (
                            lpSum([
                                self.var["omega"][model][b][
                                    self.param["sigma"][model][h]
                                ][j]
                                for j in self.param["M"]
                            ])
                            + self.var["s_star"][model][index]
                        )
                    else:
                        self.prob += lpSum([
                            self.var["omega"][model][a][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ]) == lpSum([
                            self.var["omega"][model][b][self.param["sigma"][model][h]][
                                j
                            ]
                            for j in self.param["M"]
                        ])

                    # self.prob += lpSum(
                    #     [
                    #         self.var["omega"][a][self.param["sigma"][h]][j]
                    #         for j in self.param["M"]
                    #     ]
                    # ) >= (
                    #     lpSum(
                    #         [
                    #             self.var["omega"][b][self.param["sigma"][h]][j]
                    #             for j in self.param["M"]
                    #         ]
                    #     )
                    #     + self.gamma * self.var["s_star"][index]
                    # )

                    # self.prob += lpSum(
                    #     [
                    #         self.var["omega"][b][self.param["sigma"][h]][j]
                    #         for j in self.param["M"]
                    #     ]
                    # ) >= (
                    #     lpSum(
                    #         [
                    #             self.var["omega"][a][self.param["sigma"][h]][j]
                    #             for j in self.param["M"]
                    #         ]
                    #     )
                    #     + self.gamma * self.var["s_star"][index]
                    # )

        # Constraint on refused preferences
        for index in range(len(self.preference_refused)):
            self.prob += lpSum([
                self.var["s"][self.param["c"]][preference_relations_union.index(r)][
                    self.param["sigma"][self.param["c"]][self.param["k"]]
                ]
                for r in self.preference_refused[index]
            ]) + lpSum([
                1
                - self.var["s_star"][self.param["c"]][
                    indifference_relations_union.index(r)
                ]
                for r in self.indifference_refused[index]
            ]) <= len(
                set(frozenset(r.elements) for r in self.preference_refused[index])
                | set(frozenset(r.elements) for r in self.indifference_refused[index])
            ) - self.count_refused[index] * (1 - self.var["R"][index])

        # Constraints on minimum number of preferences changes
        for dm in self.param["DM"]:
            self.prob += self.var["S"] >= self.preferences_changed[dm] + lpSum([
                1
                - self.var["s"][self.param["c"]][preference_relations_union.index(r)][
                    self.param["sigma"][self.param["c"]][self.param["k"]]
                ]
                for r in self.preference_relations[dm]
            ]) + lpSum([
                self.var["s_star"][self.param["c"]][
                    indifference_relations_union.index(r)
                ]
                for r in self.indifference_relations[dm]
            ])

        # Distance
        for dm in self.param["DM"]:
            for j in self.param["M"]:
                self.prob += (
                    self.var["W_abs"][dm][j]
                    >= self.var["w"][dm][j] - self.var["w"][self.param["c"]][j]
                )
                self.prob += (
                    self.var["W_abs"][dm][j]
                    >= self.var["w"][self.param["c"]][j] - self.var["w"][dm][j]
                )

                for h in self.param["profile_indices"]:
                    self.prob += (
                        self.var["P_abs"][dm][h][j]
                        >= self.var["p"][dm][h][j]
                        - self.var["p"][self.param["c"]][h][j]
                    )
                    self.prob += (
                        self.var["P_abs"][dm][h][j]
                        >= self.var["p"][self.param["c"]][h][j]
                        - self.var["p"][dm][h][j]
                    )

            self.prob += self.var["W"] >= lpSum([
                self.var["W_abs"][dm][j] for j in self.param["M"]
            ])
            self.prob += self.var["P"] >= lpSum([
                self.var["P_abs"][dm][h][j]
                for j in self.param["M"]
                for h in self.param["profile_indices"]
            ])

        if self.best_objective is not None:
            self.prob += (
                self.var["S"]
                + 2
                * (
                    len(preference_relations_union_indices)
                    + len(indifference_relations_union_indices)
                )
                * lpSum(self.var["R"])
                <= self.best_objective - 1
            )

    def create_solution(self):
        weights = np.array([
            value(self.var["w"][self.param["c"]][j]) for j in self.param["M"]
        ])
        profiles = NormalPerformanceTable([
            [value(self.var["p"][self.param["c"]][h][j]) for j in self.param["M"]]
            for h in self.param["profile_indices"]
        ])

        return SRMPModel(
            profiles=profiles,
            weights=weights,
            lexicographic_order=[
                p - 1 for p in self.param["sigma"][self.param["c"]][1:]
            ],
        )
