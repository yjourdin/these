"""This module implements the RMP algorithm,
as well as the preference elicitation algorithm and plot functions.

Implementation and naming conventions are taken from
:cite:p:`olteanu2022preference`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from mcda.internal.core.interfaces import Ranker
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.scales import NormalScale
from mcda.internal.core.values import Ranking
from mcda.matrices import PerformanceTable, create_outranking_matrix
from mcda.plot import (
    Annotation,
    AreaPlot,
    Axis,
    BarPlot,
    Figure,
    HorizontalStripes,
    LinePlot,
    ParallelCoordinatesPlot,
    Text,
)
from mcda.scales import DiscreteQuantitativeScale, PreferenceDirection
from mcda.transformers import ClosestTransformer
from mcda.values import Values, CommensurableValues
from pandas import DataFrame, Series, concat
from scipy.stats import rankdata


class ProfileWiseOutranking(Ranker):
    """This class infers outranking relations related to a single profile.

    The relation compares each criterion of each alternative values with the
    category profile (``1`` if better or equal, ``0`` otherwise), apply
    the `criteria_capacities` as a weighted sum for each alternative and compare
    those scores.

    :param performance_table:
    :param criteria_capacities:
    :param profile:
    """

    def __init__(
        self,
        performance_table: PerformanceTable,
        criteria_capacities: dict[frozenset[Any], float],
        profile: Values,
    ):
        self.performance_table = performance_table
        self.criteria_capacities = criteria_capacities
        self.profile = profile

    def rank(self, **kwargs) -> OutrankingMatrix:
        """Construct an outranking matrix.

        :return:
        """
        capacities = Series(
            {
                a: self.criteria_capacities[
                    frozenset(
                        [
                            c
                            for c, s in av.scales.items()
                            if s.is_better_or_equal(av[c], self.profile[c])
                        ]
                    )
                ]
                for a, av in self.performance_table.alternatives_values.items()
            }
        )

        return create_outranking_matrix(
            DataFrame(
                [
                    [
                        capacities[ai] >= capacities[aj]
                        for aj in self.performance_table.alternatives
                    ]
                    for ai in self.performance_table.alternatives
                ],
                index=self.performance_table.alternatives,
                columns=self.performance_table.alternatives,
                dtype="int64",
            )
        )


class NormalProfileWiseOutranking(ProfileWiseOutranking):
    """This class infers outranking relations related to a single profile.

    The relation compares each criterion of each alternative values with the
    category profile (``1`` if better or equal, ``0`` otherwise), apply
    the `criteria_capacities` as a weighted sum for each alternative and compare
    those scores.

    :param performance_table:
    :param criteria_capacities:
    :param profile:
    """

    def __init__(
        self,
        performance_table: PerformanceTable[NormalScale],
        criteria_capacities: dict[frozenset[Any], float],
        profile: Values[NormalScale],
    ):
        self.performance_table = performance_table
        self.criteria_capacities = criteria_capacities
        self.profile = profile

    def rank(self, **kwargs):
        """Construct an outranking matrix.

        :return:
        """
        comp_df = self.performance_table.data.ge(self.profile.data)

        capacities = np.array(
            [
                self.criteria_capacities[frozenset(np.nonzero(a)[0])]
                for a in comp_df.values
            ]
        )

        return np.greater_equal.outer(capacities, capacities)


class RMP(Ranker):
    """This class implements the RMP algorithm.

    :param performance_table:
    :param criteria_capacities:
    :param profiles:
    :param lexicographic_order: profile indices used sequentially to rank
    """

    def __init__(
        self,
        performance_table: PerformanceTable,
        criteria_capacities: dict[frozenset[Any], float],
        profiles: PerformanceTable,
        lexicographic_order: list[int],
    ):
        self.performance_table = performance_table
        self.criteria_capacities = criteria_capacities
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    @property
    def sub_rmp(self) -> list[ProfileWiseOutranking]:
        """Return list of sub RMP problems (one per category profile).

        :return:
        """
        return [
            ProfileWiseOutranking(
                self.performance_table,
                self.criteria_capacities,
                self.profiles.alternatives_values[profile],
            )
            for profile in self.profiles.alternatives
        ]

    def construct(self) -> list[OutrankingMatrix]:
        """Construct one outranking matrix per category profile.

        :return:
        """
        return [sub_rmp.rank() for sub_rmp in self.sub_rmp]

    def exploit(
        self,
        outranking_matrices: list[OutrankingMatrix],
        lexicographic_order: list[int] | None = None,
    ) -> Ranking:
        """Merge outranking matrices built by profiles in lexicographic
        order using RMP exploitation method.

        :param outranking_matrices:
            outranking matrix constructed in :attr:`profiles` order
        :param lexicographic_order: (if not supplied, use attribute)
        :return:
            the outranking total order as a ranking
        """
        lexicographic_order = (
            self.lexicographic_order
            if lexicographic_order is None
            else lexicographic_order
        )
        relations_ordered = [outranking_matrices[i] for i in lexicographic_order]
        n = len(relations_ordered)
        score = sum(
            [relations_ordered[i].data * 2 ** (n - 1 - i) for i in range(n)],
            DataFrame(
                0,
                index=relations_ordered[0].vertices,
                columns=relations_ordered[0].vertices,
            ),
        )
        outranking_matrix = score - score.transpose() >= 0
        scores = outranking_matrix.sum(1)
        scores_ordered = sorted(set(scores.values), reverse=True)
        ranks = cast(Series, scores.apply(lambda x: scores_ordered.index(x) + 1))
        return CommensurableValues(
            ranks,
            scale=DiscreteQuantitativeScale(
                ranks.tolist(),
                PreferenceDirection.MIN,
            ),
        )

    def rank(self, **kwargs) -> Ranking:
        """Compute the RMP algorithm

        :return:
            the outranking total order as a ranking
        """
        return self.exploit(self.construct())

    @classmethod
    def plot_input_data(
        cls,
        performance_table: PerformanceTable,
        rmp: RMP | None = None,
        profiles: PerformanceTable | None = None,
        lexicographic_order: list[int] | None = None,
        annotations: bool = False,
        annotations_alpha: float = 0.5,
        scales_boundaries: bool = False,
        figsize: tuple[float, float] | None = None,
        xticklabels_tilted: bool = False,
        **kwargs,
    ):  # pragma: nocover
        """Visualize input data.

        For each criterion, the arrow indicates the preference direction.

        :param performance_table:
        :param rmp: a RMP object (if given, overrides RMP parameters)
        :param criteria_capacities:
        :param profiles:
        :param lexicographic_order: profile indices used sequentially to rank
        :param annotations:
            if ``True`` every point is annotated with its value
        :param annotations_alpha: annotations white box transparency
        :param scales_boundaries:
            if ``True`` the criteria boundaries are the scales boundaries,
            else they are computed from the data
        :param figsize: figure size in inches as a tuple (`width`, `height`)
        :param xticklabels_tilted:
            if ``True`` `xticklabels` are tilted to better fit
        """
        # Reorder scales
        scales = {
            crit: performance_table.scales[crit] for crit in performance_table.criteria
        }

        if rmp is not None:
            profiles = rmp.profiles
            lexicographic_order = rmp.lexicographic_order

        # Concatenate profiles with performance_table
        if profiles is not None:
            df = concat([performance_table.data, profiles.data])
        else:
            df = performance_table.data.copy()
        table = PerformanceTable(df, scales=scales)
        table = cast(PerformanceTable, table.to_numeric)
        if not scales_boundaries:
            _scales = table.scales
            table.scales = table.bounds
            # Conserve preference direction
            for key, scale in _scales.items():
                table.scales[key].preference_direction = scale.preference_direction
        table = ClosestTransformer.normalize(table)

        # Create constants
        nb_alt = len(performance_table.alternatives)
        nb_profiles = len(profiles.alternatives) if profiles is not None else 0

        # Create figure and axis
        fig = Figure(figsize=figsize)
        ax = fig.create_add_axis()

        # Axis parameters
        x = cast(list[float], range(len(performance_table.criteria)))
        xticks = cast(list[float], (range(len(performance_table.criteria))))
        xticklabels = [f"{crit}" for crit in performance_table.criteria]

        # Plotted annotations' coordinates
        annotations_coord: list[tuple[float, float]] = []

        # Profiles
        if profiles is not None:
            for profile in range(nb_alt, nb_alt + nb_profiles):
                ax.add_plot(
                    AreaPlot(
                        x,
                        table.data.iloc[profile].to_list(),
                        xticks=xticks,
                        yticks=[],
                        xticklabels=xticklabels,
                        xticklabels_tilted=xticklabels_tilted,
                        color="black",
                        alpha=0.1,
                        strongline=False,
                    )
                )
                ax.add_plot(
                    Annotation(
                        0,
                        cast(float, table.data.iloc[profile, 0]),
                        f"$P^{profile - nb_alt}$",
                        -1,
                        0,
                        "right",
                        "center",
                    )
                )

        # Alternatives
        values = table.data[:nb_alt]
        labels = table.data[:nb_alt].index
        ax.add_plot(
            ParallelCoordinatesPlot(
                x,
                values,
                xticks=xticks,
                yticks=[],
                xticklabels=xticklabels,
                xticklabels_tilted=xticklabels_tilted,
                labels=labels,
                linestyle="-.",
            )
        )
        # Legend
        ax.add_legend(title="Alternatives :", location="right")

        fig.draw()
        assert ax.ax is not None  # to comply with mypy

        # Annotations
        if annotations:
            if profiles is not None:
                for profile in range(nb_alt, nb_alt + nb_profiles):
                    for i in x:
                        i = cast(int, i)
                        xy = (i, table.data.iloc[profile, i])
                        overlap = False
                        for xc, yc in annotations_coord:
                            if (xc == i) and (
                                abs(
                                    ax.ax.transData.transform(xy)[1]
                                    - ax.ax.transData.transform((xc, yc))[1]
                                )
                                < 20
                            ):
                                # if current annotation overlaps
                                # already plotted annotations
                                overlap = True
                                break

                        if not overlap:
                            annotation = Annotation(
                                i,
                                cast(float, table.data.iloc[profile, i]),
                                cast(str, profiles.data.iloc[profile - nb_alt, i]),
                                2,
                                0,
                                "left",
                                "center",
                                annotations_alpha,
                            )
                            ax.add_plot(annotation)
                            annotations_coord.append(
                                (
                                    cast(float, i),
                                    cast(float, table.data.iloc[profile, i]),
                                )
                            )

            for alt in range(nb_alt):
                for i in x:
                    i = cast(int, i)
                    xy = (i, table.data.iloc[alt, i])
                    overlap = False
                    for xc, yc in annotations_coord:
                        if (xc == i) and (
                            abs(
                                ax.ax.transData.transform(xy)[1]
                                - ax.ax.transData.transform((xc, yc))[1]
                            )
                            < 20
                        ):
                            # if current annotation overlaps
                            # already plotted annotations
                            overlap = True
                            break

                    if not overlap:
                        annotation = Annotation(
                            i,
                            cast(float, table.data.iloc[alt, i]),
                            cast(str, performance_table.data.iloc[alt, i]),
                            2,
                            0,
                            "left",
                            "center",
                            annotations_alpha,
                        )
                        ax.add_plot(annotation)
                        annotations_coord.append(
                            (cast(float, i), cast(float, table.data.iloc[alt, i]))
                        )

        # Lexicographic order
        if lexicographic_order is not None:
            text = Text(
                0,
                1.2,
                "Lexicographic order : $"
                + r" \rightarrow ".join(
                    [f"P^{profile}" for profile in lexicographic_order]
                )
                + "$",
                box=True,
            )
            ax.add_plot(text)
        fig.draw()

    def plot_concordance_index(
        self,
        performance_table: PerformanceTable,
        figsize: tuple[float, float] | None = None,
        ncols: int = 0,
        nrows: int = 0,
        xlabels_tilted: bool = False,
        **kwargs,
    ):  # pragma: nocover
        """Visualize concordance index between alternatives and profiles

        :param performance_table:
        :param figsize: figure size in inches as a tuple (`width`, `height`)
        :param xlabels_tilted:
            if ``True`` `xlabels` are tilted to better fit
        """
        # Create constants
        nb_alt = len(performance_table.alternatives)
        nb_profiles = len(self.profiles.alternatives)
        capacities_sum = self.criteria_capacities[
            frozenset(performance_table.data.columns)
        ]

        # Create figure and axes
        fig = Figure(figsize=figsize, ncols=ncols, nrows=nrows)

        for ind_alt in range(nb_alt):
            ax = Axis(
                xlabel=f"{performance_table.data.index[ind_alt]}",
                xlabel_tilted=xlabels_tilted,
            )
            # Axis properties
            x = cast(list[float], range(nb_profiles))
            xticks = cast(list[float], range(nb_profiles))
            xticklabels = [f"$P^{profile}$" for profile in self.lexicographic_order]
            ylim = (0.0, 1.0)

            values = [
                self.criteria_capacities[
                    frozenset(
                        [
                            crit
                            for ind_crit, crit in enumerate(performance_table.criteria)
                            if performance_table.scales[crit].is_better_or_equal(
                                performance_table.data.iloc[ind_alt, ind_crit],
                                self.profiles.data.iloc[profile, ind_crit],
                            )
                        ]
                    )
                ]
                / capacities_sum
                for profile in self.lexicographic_order
            ]
            ax.add_plot(
                BarPlot(
                    x,
                    values,
                    ylim=ylim,
                    xticks=xticks,
                    xticklabels=xticklabels,
                )
            )
            fig.add_axis(ax)
        fig.axes[-1].add_legend(title="Criteria :", location="right")
        fig.draw()

    def plot_progressive_ranking(
        self,
        performance_table: PerformanceTable,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ):  # pragma: nocover
        """Visualize ranking progressively according to the lexicographic order

        :param performance_table:
        :param figsize: figure size in inches as a tuple (`width`, `height`)
        """
        # Create constants
        nb_alt = len(performance_table.alternatives)
        nb_profiles = len(self.lexicographic_order)

        # Compute rankings progressively
        relations = self.construct()
        rankings = DataFrame(
            [
                self.exploit(relations, self.lexicographic_order[:stop]).data
                for stop in range(1, nb_profiles + 1)
            ]
        )

        # Compute ranks
        final_values = rankings.iloc[nb_profiles - 1].drop_duplicates().sort_values()
        value_to_rank = {
            value: rank
            for value, rank in zip(final_values, range(1, len(final_values) + 1))
        }
        ranks = rankings.map(lambda x: value_to_rank[x])
        nb_ranks = len(value_to_rank)

        # Create figure and axes
        fig = Figure(figsize=figsize)
        ax = Axis(xlabel="Profiles", ylabel="Rank")
        fig.add_axis(ax)

        # Axis parameters
        xticks = cast(list[float], range(nb_profiles))
        xticklabels = [f"$P^{profile}$" for profile in self.lexicographic_order]
        ylim = (0.5, nb_ranks + 0.5)
        yticks = cast(list[float], range(1, nb_ranks + 1))
        yminorticks = np.arange(1, nb_ranks + 2) - 0.5
        yticklabels = cast(list[str], range(nb_ranks, 0, -1))

        # Draw horizontal striped background
        ax.add_plot(
            HorizontalStripes(
                yminorticks.tolist(),
                color="black",
                alpha=0.1,
                attach_yticks=True,
            )
        )

        # Number of alternatives for each rank (depending on the profile)
        rank_counts = DataFrame(
            [
                {
                    k: v
                    for k, v in zip(*np.unique(ranks.loc[profile], return_counts=True))
                }
                for profile in ranks.index
            ],
            columns=range(1, nb_alt + 1),
        ).fillna(0)
        # Offsets' width for each rank (depending on the profile)
        offsets_width = 1 / (rank_counts + 1)
        # Offsets to apply to current alternative's ranks
        offsets = [0.5] * nb_profiles
        # Alternatives sorted according to the final ranking
        final_ranking_sorted = rankings.iloc[-1].sort_values(ascending=False).index
        # Previous alternative's ranks
        previous_ranks = [0] * nb_profiles

        for alt in final_ranking_sorted:
            # Current alternative's ranks
            current_ranks = ranks[alt]
            # Update offsets (return to 0.5 if it's a new rank)
            offsets = np.where(current_ranks == previous_ranks, offsets, 0.5).tolist()
            offsets = [
                offsets[profile] - offsets_width.loc[profile, current_ranks[profile]]
                for profile in range(nb_profiles)
            ]
            x = cast(list[float], range(nb_profiles))
            y = current_ranks + offsets
            ax.add_plot(
                LinePlot(
                    x,
                    y,
                    xticks=xticks,
                    xticklabels=xticklabels,
                    ylim=ylim,
                    yticks=yticks,
                    yticklabels=yticklabels,
                    marker="o",
                )
            )
            ax.add_plot(
                Annotation(
                    nb_profiles - 1,
                    current_ranks.iloc[-1] + offsets[-1],
                    str(alt),
                    10,
                    0,
                    vertical_alignement="center",
                    box=True,
                )
            )
            previous_ranks = current_ranks
        fig.draw()


class NormalRMP(RMP):
    def __init__(
        self,
        performance_table: PerformanceTable[NormalScale],
        criteria_capacities: dict[frozenset[Any], float],
        profiles: PerformanceTable[NormalScale],
        lexicographic_order: list[int],
    ):
        self.performance_table = performance_table
        self.criteria_capacities = criteria_capacities
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    @property
    def sub_rmp(self):
        """Return list of sub RMP problems (one per category profile).

        :return:
        """
        return [
            NormalProfileWiseOutranking(
                self.performance_table,
                self.criteria_capacities,
                self.profiles.alternatives_values[profile],
            )
            for profile in self.profiles.alternatives
        ]

    def rank(self, **kwargs):
        """Compute the RMP algorithm

        :return:
            the outranking total order as a ranking
        """
        profilewise_outranking_matrices = np.array(
            [sub_rmp.rank() for sub_rmp in self.sub_rmp]
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
