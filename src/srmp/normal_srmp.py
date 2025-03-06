from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from mcda import PerformanceTable
from mcda.internal.core.scales import NormalScale
from mcda.outranking.srmp import SRMP, ProfileWiseOutranking
from mcda.values import Values
from scipy.stats import rankdata

OutrankingMatrix = npt.NDArray[np.bool_]


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
        criteria_weights: npt.NDArray[np.float64],
        profile: Values[NormalScale],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profile = profile

    def rank(self, **kwargs: Any):  # type: ignore
        """Construct an outranking matrix.

        :return:
        """
        conditional_weighted_sum: npt.NDArray[np.float64] = np.dot(
            self.performance_table.data.values >= self.profile.data.values,
            self.criteria_weights,
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
        criteria_weights: npt.NDArray[np.float64],
        profiles: PerformanceTable[NormalScale],
        lexicographic_order: list[int],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    @property
    def sub_srmp(self) -> Sequence[NormalProfileWiseOutranking]:  # type: ignore
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

    def rank_numpy(self):
        profilewise_outranking_matrices = np.array([
            sub_srmp.rank() for sub_srmp in self.sub_srmp
        ])
        relations_ordered = [
            profilewise_outranking_matrices[i] for i in self.lexicographic_order
        ]
        n = len(relations_ordered)
        power = np.array([2 ** (n - 1 - i) for i in range(n)])
        score = np.sum(relations_ordered * power[:, None, None], 0)
        outranking_matrix = score - score.transpose() >= 0
        scores = outranking_matrix.sum(1)
        return rankdata(-scores, method="dense").astype(np.int_)
