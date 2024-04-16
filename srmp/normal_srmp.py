from typing import Any

import numpy as np
from mcda.matrices import PerformanceTable
from mcda.scales import DiscreteQuantitativeScale, PreferenceDirection
from mcda.internal.core.scales import NormalScale
from mcda.values import Values, CommensurableValues
from mcda.outranking.srmp import SRMP, ProfileWiseOutranking
from pandas import Series
from scipy.stats import rankdata

OutrankingMatrix = np.ndarray


class NormalProfileWiseOutranking(ProfileWiseOutranking):
    """This class infers outranking relations related to a single profile.

    The relation compares each criterion of each alternative values with the
    category profile (``1`` if better or equal, ``0`` otherwise), apply
    the `criteria_weights` as a weighted sum for each alternative and compare
    those scores.

    :param performance_table:
    :param criteria_weights:
    :param profile:
    """

    def __init__(
        self,
        performance_table: PerformanceTable[NormalScale],
        criteria_weights: dict[Any, float],
        profile: Values[NormalScale],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profile = profile

    def rank(self, **kwargs):
        """Construct an outranking matrix.

        :return:
        """
        conditional_weighted_sum = np.dot(
            self.performance_table.data.values >= self.profile.data.values,
            np.array(list(self.criteria_weights.values())),
        )

        return np.greater_equal.outer(
            conditional_weighted_sum, conditional_weighted_sum
        )


class NormalSRMP(SRMP):
    """This class implements the SRMP algorithm with a NormalPerformanceTable.

    :param performance_table:
    :param criteria_weights:
    :param profiles:
    :param lexicographic_order: profile indices used sequentially to rank
    """

    def __init__(
        self,
        performance_table: PerformanceTable[NormalScale],
        criteria_weights: dict[Any, float],
        profiles: PerformanceTable[NormalScale],
        lexicographic_order: list[int],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    @property
    def sub_srmp(self):
        """Return list of sub SRMP problems (one per category profile).

        :return:
        """
        return [
            NormalProfileWiseOutranking(
                self.performance_table,
                self.criteria_weights,
                self.profiles.alternatives_values[profile],
            )
            for profile in self.profiles.alternatives
        ]

    def rank(self, **kwargs):
        """Compute the SRMP algorithm

        :return:
            the outranking total order as a ranking
        """
        profilewise_outranking_matrices = np.array(
            [sub_srmp.rank() for sub_srmp in self.sub_srmp]
        )
        relations_ordered = [
            profilewise_outranking_matrices[i] for i in self.lexicographic_order
        ]
        n = len(relations_ordered)
        power = np.array([2 ** (n - 1 - i) for i in range(n)])
        score = np.sum(relations_ordered * power[:, None, None], 0)
        outranking_matrix = score - score.transpose() >= 0
        scores = outranking_matrix.sum(1)
        ranks = rankdata(-scores, method="dense")
        return CommensurableValues(
            Series(ranks, self.performance_table.alternatives),
            scale=DiscreteQuantitativeScale(
                ranks.tolist(),
                PreferenceDirection.MIN,
            ),
        )
