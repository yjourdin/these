from collections.abc import Sequence
from typing import Any

import numpy as np
from mcda.internal.core.scales import NormalScale
from mcda.outranking.srmp import SRMP, ProfileWiseOutranking
from mcda.values import Values
from scipy.stats import rankdata

from src.performance_table.normal_performance_table import NormalPerformanceTable

OutrankingMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool]]


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
        performance_table: NormalPerformanceTable,
        criteria_weights: np.ndarray[tuple[int], np.dtype[np.float64]],
        profile: Values[NormalScale],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profile = profile

    def rank_numpy(self, **kwargs: Any):
        """Construct an outranking matrix.

        :return:
        """
        conditional_weighted_sum: np.ndarray[tuple[int], np.dtype[np.float64]] = np.dot(
            self.performance_table.data.values >= self.profile.data.values,
            self.criteria_weights,
        )

        return np.greater_equal.outer(
            conditional_weighted_sum, conditional_weighted_sum
        ).astype(dtype=np.bool)

    def credibility_numpy(self, **kwargs: Any):
        """Construct an outranking matrix.

        :return:
        """
        comp_df = self.performance_table.data.values >= self.profile.data.values

        comp_bit = np.packbits(comp_df, axis=1, bitorder="little").squeeze()

        conditional_weighted_sum = np.dot(comp_df, self.criteria_weights)

        result = np.subtract.outer(conditional_weighted_sum, conditional_weighted_sum)

        result[result == 0] = np.dot(
            np.unpackbits(
                (~np.bitwise_xor.outer(comp_bit, comp_bit))[:, :, np.newaxis],
                axis=-1,
                count=len(self.performance_table.criteria),
            ),
            self.criteria_weights,
        )[result == 0]

        return result, np.equal.outer(conditional_weighted_sum, conditional_weighted_sum)


class NormalSRMP(SRMP):
    """This class implements the SRMP algorithm with a NormalPerformanceTable.

    :param performance_table:
    :param criteria_weights:
    :param profiles:
    :param lexicographic_order: profile indices used sequentially to rank
    """

    def __init__(
        self,
        performance_table: NormalPerformanceTable,
        criteria_weights: np.ndarray[tuple[int], np.dtype[np.float64]],
        profiles: NormalPerformanceTable,
        lexicographic_order: list[int],
    ):
        self.performance_table = performance_table
        self.criteria_weights = criteria_weights
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    @property
    def sub_srmp(self) -> Sequence[NormalProfileWiseOutranking]:  # pyright: ignore[reportIncompatibleMethodOverride]
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
            sub_srmp.rank_numpy() for sub_srmp in self.sub_srmp
        ])
        relations_ordered = profilewise_outranking_matrices[self.lexicographic_order]
        n = len(relations_ordered)
        power = 2 ** (n - 1 - np.arange(n))
        score = np.sum(relations_ordered * power[:, None, None], 0)
        outranking_matrix = score - score.transpose() >= 0
        scores = outranking_matrix.sum(1)
        return np.array(rankdata(-scores, method="dense")).astype(np.int_)  # pyright: ignore[reportUnknownArgumentType]

    def credibility_numpy(self):
        credibility_matrices, equality_matrices = np.rollaxis(
            np.array([sub_srmp.credibility_numpy() for sub_srmp in self.sub_srmp]), 1
        )
        equality_matrices = equality_matrices.astype(np.bool_, copy=False)
        credibility_ordered = credibility_matrices[self.lexicographic_order]
        equality_ordered = equality_matrices[self.lexicographic_order]
        n = len(self.lexicographic_order)
        score = np.ones_like(credibility_ordered[0])
        i = 0
        eq = equality_ordered[i]
        prev_eq = np.ones_like(eq, np.bool_)
        while (i < n) and prev_eq.any(axis=None):
            score[prev_eq & ~eq] = credibility_ordered[i][prev_eq & ~eq]
            score[prev_eq & eq] = np.minimum(
                score[prev_eq & eq], credibility_ordered[i][prev_eq & eq]
            )
            i += 1
            if i < n:
                prev_eq = eq
                eq = equality_ordered[i]
        return score
