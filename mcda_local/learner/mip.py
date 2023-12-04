"""This module implements the SRMP algorithm,
as well as the preference elicitation algorithm and plot functions.

Implementation and naming conventions are taken from
:cite:p:`olteanu2022preference`.
"""
from itertools import permutations

from pulp import LpBinary, LpContinuous, LpMaximize, LpProblem, LpVariable, lpSum, value

from ..core.learner import Learner
from ..core.performance_table import PerformanceTable
from ..core.relations import (
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
)
from ..core.scales import NormalScale
from ..ranker.srmp import SRMP


def pulp_value(x) -> float:
    return value(x)


class MIP(Learner[SRMP]):
    """This class gathers functions used to learn a SRMP model.

    :param performance_table:
    :param relations:
    :param max_profiles_number: highest number of reference profiles
    :param profiles_number: number of reference profiles
    :param lexicographic_order: profile indices used sequentially to rank
    :param inconsistencies:
        if ``True`` inconsistent comparisons will be taken into account
    :param gamma: value used for modeling strict inequalities
    :param non_dictator: if ``True`` prevent dictator weights (> 0.5)
    :param solver_args: extra arguments supplied to the solver
    :raise TypeError:
        if `max_profiles_number`, `profiles_number`
        and `lexicographic_order` are not specified

    .. note::
        If multiple arguments are supplied, only one will be used in the
        following priority: `lexicographic_order`, `profiles_number` then
        `max_profiles_number`
    """

    def __init__(
        self,
        max_profiles_number: int | None = None,
        profiles_number: int | None = None,
        lexicographic_order: list[int] | None = None,
        inconsistencies: bool = True,
        gamma: float = 0.001,
        non_dictator: bool = False,
        solver_args: dict | None = None,
    ):
        # Check parameters provided
        provided = (
            (max_profiles_number is not None)
            + (profiles_number is not None)
            + (lexicographic_order is not None)
        )
        if provided == 0:
            raise ValueError(
                "You must specify either 'max_profiles_number',\
                'profiles_number' or 'lexicographic_order'"
            )
        self.max_profiles_number = max_profiles_number
        self.profiles_number = profiles_number
        self.lexicographic_order = lexicographic_order
        self.inconsistencies = inconsistencies
        self.gamma = gamma
        self.non_dictator = non_dictator
        self.solver_args = solver_args or {}

    def fitness(self, problem: LpProblem, nb_relations: int) -> float:
        """Compute fitness of a SRMP solution.

        :param problem: LP problem (solved)
        :param nb_relations: number of relations supplied for learning
        :param inconsistencies:
            if ``True`` inconsistent comparisons will be taken into account
        """
        return (
            0
            if problem.status != 1
            else (
                pulp_value(problem.objective) / nb_relations
                if self.inconsistencies
                else 1
            )
        )

    def _learn_lexicographic_order(
        self,
        performance_table: PerformanceTable,
        relations: PreferenceStructure,
        lexicographic_order: list[int],
    ) -> tuple[SRMP | None, LpProblem]:
        """Train a SRMP instance using lexicographic order

        :param lexicographic_order: profile indices used sequentially to rank
        :return: the inferred SRMP object, along its LP problem
        """
        performance_table = performance_table.normalize()

        ##############
        # Parameters #
        ##############

        # list of alternatives
        A_star = relations.elements
        # list of criteria
        M = performance_table.criteria
        # Number of profiles
        k = len(lexicographic_order)
        # Indices of profiles
        profile_indices = list(range(1, k + 1))
        # Lexicographic order
        sigma = [0] + [profile + 1 for profile in lexicographic_order]
        # Binary comparisons with preference
        preference_relations = PreferenceStructure(relations[PreferenceRelation])
        preference_relations_indices = range(len(preference_relations))
        # Binary comparisons with indifference
        indifference_relations = PreferenceStructure(relations[IndifferenceRelation])
        indifference_relations_indices = range(len(indifference_relations))

        #############
        # Variables #
        #############

        # Weights
        w = LpVariable.dicts("Weight", M, lowBound=0, upBound=1, cat=LpContinuous)
        # Reference profiles
        p = LpVariable.dicts("Profile", (profile_indices, M), cat=LpContinuous)
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
            cat=LpContinuous,
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
        # to comply with Pylance (reportUnboundVariable)
        else:
            s_star = LpVariable.dicts("IndifferenceRankingVariable")

        ##############
        # LP problem #
        ##############

        prob = LpProblem("SRMP_Elicitation", LpMaximize)

        if self.inconsistencies:
            prob += lpSum(
                [s[index][0] for index in preference_relations_indices]
            ) + lpSum([s_star[index] for index in indifference_relations_indices])

        ###############
        # Constraints #
        ###############

        # Normalized weights
        prob += lpSum([w[j] for j in M]) == 1

        for j in M:
            if self.non_dictator:
                # Non-dictator weights
                prob += w[j] <= 0.5

            # Non-zero weights
            prob += w[j] >= self.gamma

            # Constraints on the reference profiles
            prob += p[1][j] >= 0
            prob += p[k][j] <= 1

            for h in profile_indices:
                if h != k:
                    # Dominance between the reference profiles
                    prob += p[h + 1][j] >= p[h][j]

                for a in A_star:
                    # Constraints on the local concordances
                    prob += (
                        performance_table.data.loc[a, j] - p[h][j] >= delta[a][h][j] - 1
                    )
                    prob += (
                        delta[a][h][j]
                        >= performance_table.data.loc[a, j] - p[h][j] + self.gamma
                    )

                    # Constraints on the weighted local concordances
                    prob += omega[a][h][j] <= w[j]
                    prob += omega[a][h][j] >= 0
                    prob += omega[a][h][j] <= delta[a][h][j]
                    prob += omega[a][h][j] >= delta[a][h][j] + w[j] - 1

        # Constraints on the preference ranking variables
        for index in preference_relations_indices:
            if not self.inconsistencies:
                prob += s[index][sigma[0]] == 1
            prob += s[index][sigma[k]] == 0

        for h in profile_indices:
            # Constraints on the preferences
            for index, relation in enumerate(preference_relations):
                a, b = relation.a, relation.b

                prob += lpSum([omega[a][sigma[h]][j] for j in M]) >= (
                    lpSum([omega[b][sigma[h]][j] for j in M])
                    + self.gamma
                    - s[index][sigma[h]] * (1 + self.gamma)
                    - (1 - s[index][sigma[h - 1]])
                )

                prob += lpSum([omega[a][sigma[h]][j] for j in M]) >= (
                    lpSum([omega[b][sigma[h]][j] for j in M])
                    - (1 - s[index][sigma[h]])
                    - (1 - s[index][sigma[h - 1]])
                )

                prob += lpSum([omega[a][sigma[h]][j] for j in M]) <= (
                    lpSum([omega[b][sigma[h]][j] for j in M])
                    + (1 - s[index][sigma[h]])
                    + (1 - s[index][sigma[h - 1]])
                )

            # Constraints on the indifferences
            for index, relation in enumerate(indifference_relations):
                a, b = relation.a, relation.b
                if not self.inconsistencies:
                    prob += lpSum([omega[a][sigma[h]][j] for j in M]) == lpSum(
                        [omega[b][sigma[h]][j] for j in M]
                    )
                else:
                    prob += lpSum([omega[a][sigma[h]][j] for j in M]) <= (
                        lpSum([omega[b][sigma[h]][j] for j in M]) - (1 - s_star[index])
                    )

                    prob += lpSum([omega[b][sigma[h]][j] for j in M]) <= (
                        lpSum([omega[a][sigma[h]][j] for j in M]) - (1 - s_star[index])
                    )

        # Solve problem
        status = prob.solve(**self.solver_args)

        if status != 1:
            return None, prob

        # Compute optimum solution
        criteria_weights = {j: pulp_value(w[j]) for j in M}
        profiles = PerformanceTable(
            [[pulp_value(p[h][j]) for j in M] for h in profile_indices],
            criteria=M,
            scales={c: NormalScale() for c in M},
        )
        # Denormalize profile values
        profiles = profiles.transform(performance_table.scales)

        return SRMP(criteria_weights, profiles, lexicographic_order), prob

    def _learn(
        self,
        performance_table: PerformanceTable,
        relations: PreferenceStructure,
        lexicographic_order: list[int] | None = None,
        profiles_number: int | None = None,
        max_profiles_number: int | None = None,
    ) -> tuple[SRMP | None, LpProblem]:
        """Learn a SRMP instance

        :param lexicographic_order: profile indices used sequentially to rank
        :param profiles_number: number of reference profiles
        :param max_profiles_number: highest number of reference profiles
        :return: the inferred SRMP object, along with its fitness
        :raise TypeError:
            * if `max_profiles_number`, `profiles_number` and
              `lexicographic_order` are not specified

        .. note::
            If multiple arguments are supplied, only one will be used in the
            following priority: `lexicographic_order`, `profiles_number` then
            `max_profiles_number`
        """
        # Check parameters provided
        provided = (
            (max_profiles_number is not None)
            + (profiles_number is not None)
            + (lexicographic_order is not None)
        )
        if provided == 0:  # pragma: nocover
            raise ValueError(
                "You must specify either 'max_profiles_number',\
                'profiles_number' or 'lexicographic_order'"
            )
        if lexicographic_order:
            # Compute the learning algorithm
            result, prob = self._learn_lexicographic_order(
                performance_table, relations, lexicographic_order
            )
            return result, prob
        if profiles_number:
            lexicographic_order_list = list(permutations(range(profiles_number)))
            best_result = None
            best_prob = LpProblem()
            best_fitness = -1.0
            for current_lexicographic_order in lexicographic_order_list:
                # Compute the learning algorithm for each lexicographic order
                result, prob = self._learn(
                    performance_table,
                    relations,
                    lexicographic_order=list(current_lexicographic_order),
                )
                fitness = self.fitness(prob, len(relations))

                if fitness > best_fitness:
                    best_result = result
                    best_prob = prob
                    best_fitness = fitness
                if best_fitness == 1:
                    # Break recursion when a perfect solution is found
                    break
            return best_result, best_prob

        profiles_number_list = (
            list(range(1, max_profiles_number + 1))
            if max_profiles_number is not None
            else []
        )

        best_result = None
        best_prob = LpProblem()
        best_fitness = -1.0
        for profiles_number in profiles_number_list:
            # Compute the learning algorithm for each profiles number
            result, prob = self._learn(
                performance_table,
                relations,
                profiles_number=profiles_number,
            )
            fitness = self.fitness(prob, len(relations))
            if fitness > best_fitness:
                best_result = result
                best_fitness = fitness
                best_prob = prob
            if best_fitness == 1:
                # Break recursion when a perfect solution is found
                break
        return best_result, best_prob

    def learn(
        self, train_data: PerformanceTable, target: PreferenceStructure
    ) -> SRMP | None:
        """Learn and return SRMP solution (if existing).

        :return:
        """
        result, _ = self._learn(
            train_data,
            target,
            self.lexicographic_order,
            self.profiles_number,
            self.max_profiles_number,
        )
        return result
