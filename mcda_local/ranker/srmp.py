"""This module implements the SRMP algorithm,
as well as the preference elicitation algorithm and plot functions.

Implementation and naming conventions are taken from
:cite:p:`olteanu2022preference`.
"""
from typing import Any, cast

from mcda.core.aliases import Function
from mcda.core.relations import OutrankingMatrix
from numpy import arange, array, unique, where
from pandas import DataFrame, Series, concat

from ..core.criteria_functions import CriteriaFunctions
from ..core.performance_table import PerformanceTable
from ..core.ranker import Ranker
from ..core.scales import (
    PreferenceDirection,
    QualitativeScale,
    QuantitativeScale,
    Scale,
    is_better_or_equal,
)
from ..core.values import Ranking, ScaleValues
from ..plot.plot import (
    Annotation,
    AreaPlot,
    Axis,
    BarPlot,
    Figure,
    HorizontalStripes,
    LinePlot,
    ParallelCoordinatesPlot,
    StackedBarPlot,
    Text,
)


class ProfileWiseOutranking(Ranker):
    """This class infers outranking relations related to a single profile.

    The relation compares each criterion of each alternative values with the
    category profile (``1`` if better or equal, ``0`` otherwise), apply
    the `criteria_weights` as a weighted sum for each alternative and compare
    those scores.

    :param criteria_weights:
    :param profile:

    .. todo:: Find a better name
    """

    def __init__(self, criteria_weights: dict[Any, float], profile: ScaleValues):
        self.criteria_weights = criteria_weights
        self.profile = profile

    def copy(self):
        return ProfileWiseOutranking(self.criteria_weights, self.profile)

    def construct(self, performance_table: PerformanceTable) -> OutrankingMatrix:
        """Construct the outranking matrix.

        :param performance_table:
        :return:
        """
        _profile = self.profile.transform(performance_table.scales)
        functions = CriteriaFunctions(
            {
                criterion: (
                    cast(
                        Function,
                        lambda x, c=criterion: self.criteria_weights[c]
                        if is_better_or_equal(
                            x,
                            _profile.data[c],
                            cast(QuantitativeScale, performance_table.scales[c]),
                        )
                        else 0,
                    )
                )  # https://bugs.python.org/issue13652
                for criterion in performance_table.criteria
            }
        )

        conditional_weighted_sum = cast(
            Series,
            performance_table.apply(functions).sum(1),
        )

        return OutrankingMatrix(
            DataFrame(
                [
                    [
                        conditional_weighted_sum[ai] >= conditional_weighted_sum[aj]
                        for aj in performance_table.alternatives
                    ]
                    for ai in performance_table.alternatives
                ],
                index=performance_table.alternatives,
                columns=performance_table.alternatives,
                dtype="int64",
            )
        )

    def rank(self, performance_table: PerformanceTable, **kwargs) -> OutrankingMatrix:
        """Construct an outranking matrix.

        :param performance_table:
        :return:
        """
        return self.construct(performance_table=performance_table)


class SRMP(Ranker):
    """This class implements the SRMP algorithm.

    :param criteria_weights:
    :param profiles:
    :param lexicographic_order: profile indices used sequentially to rank
    """

    def __init__(
        self,
        criteria_weights: dict[Any, float],
        profiles: PerformanceTable,
        lexicographic_order: list[int],
    ):
        self.criteria_weights = criteria_weights
        self.profiles = profiles
        self.lexicographic_order = lexicographic_order

    def copy(self) -> "SRMP":
        """Copy the SRMP model

        Returns:
            SRMP: the copy
        """
        return SRMP(self.criteria_weights, self.profiles, self.lexicographic_order)

    @property
    def sub_srmp(self) -> list[ProfileWiseOutranking]:
        """Return list of sub SRMP problems (one per category profile).

        :return:
        """
        return [
            ProfileWiseOutranking(
                self.criteria_weights,
                self.profiles.alternatives_values[profile],
            )
            for profile in self.profiles.alternatives
        ]

    def construct(self, performance_table: PerformanceTable) -> list[OutrankingMatrix]:
        """Construct one outranking matrix per category profile.

        :param performance_table:
        :return:
        """
        return [sub_srmp.rank(performance_table) for sub_srmp in self.sub_srmp]

    def exploit(
        self,
        outranking_matrices: list[OutrankingMatrix],
        lexicographic_order: list[int] | None = None,
    ) -> Ranking:
        """Merge outranking matrices built by profiles in lexicographic
        order using SRMP exploitation method.

        :param outranking_matrices:
            outranking matrix constructed in :attr:`profiles` order
        :param lexicographic_order: (if not supplied, use attribute)
        :return:
            the outranking total order as a ranking
        """
        lexicographic_order = lexicographic_order or self.lexicographic_order
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
        return Ranking(
            scores.apply(lambda x: scores_ordered.index(x) + 1),
            PreferenceDirection.MIN,
        )

    def rank(self, performance_table: PerformanceTable, **kwargs) -> Ranking:
        """Compute the SRMP algorithm

        :param performance_table:
        :return:
            the outranking total order as a ranking
        """
        return self.exploit(self.construct(performance_table))

    def plot_input_data(
        self,
        performance_table: PerformanceTable | None = None,
        annotations: bool = False,
        annotations_alpha: float = 0.5,
        scales_boundaries: bool = False,
        figsize: tuple[float, float] | None = None,
        xticklabels_tilted: bool = False,
        **kwargs,
    ):  # pragma: nocover
        """Visualize input data.

        For each criterion, the arrow indicates the preference direction.
        The criteria weights are displayed as a bar plot,
        and their values are written in parentheses

        :param performance_table:
        :param srmp: a SRMP object (if given, overrides SRMP parameters)
        :param criteria_weights:
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
        # Reorder scales and criteria_weights
        profiles = self.profiles
        scales = profiles.scales
        criteria = profiles.criteria
        criteria_weights = self.criteria_weights
        lexicographic_order = self.lexicographic_order

        # Transform to quantitative scales
        quantitative_scales = {}
        for key, scale in scales.items():
            if isinstance(scale, QualitativeScale):
                quantitative_scales[key] = scale.quantitative
            else:
                quantitative_scales[key] = cast(QuantitativeScale, scale)

        # Concatenate profiles with performance_table
        df = (
            concat([performance_table.data, profiles.data], ignore_index=True)
            if performance_table
            else profiles.data
        )
        table = PerformanceTable(df, scales=scales)
        table = table.transform(cast(dict[Any, Scale], quantitative_scales))
        if not scales_boundaries:
            table.scales = table.bounds
            # Conserve preference direction
            for key, scale in quantitative_scales.items():
                cast(
                    QuantitativeScale, table.scales[key]
                ).preference_direction = scale.preference_direction
        table = table.normalize()

        # Create constants
        nb_alt = len(performance_table.alternatives) if performance_table else 0
        nb_profiles = len(profiles.alternatives)

        # Create figure and axis
        fig = Figure(figsize=figsize)
        ax = fig.create_add_axis()

        # Axis parameters
        x = cast(list[float], range(len(criteria)))
        xticks = cast(list[float], (range(len(criteria))))
        xticklabels = [
            f"{crit}\n({weight})" for crit, weight in criteria_weights.items()
        ]

        # Criteria weights
        weights = array([*criteria_weights.values()])
        weights_normalized = weights / weights.sum()
        ax.add_plot(
            BarPlot(
                x,
                weights_normalized,
                xticks=xticks,
                yticks=[],
                xticklabels=xticklabels,
                xticklabels_tilted=xticklabels_tilted,
                width=0.1,
                alpha=0.5,
            )
        )

        # Plotted annotations' coordinates
        annotations_coord: list[tuple[float, float]] = []

        # Profiles
        for profile in range(nb_alt, nb_alt + nb_profiles):
            ax.add_plot(
                AreaPlot(
                    x,
                    cast(list[float], table.data.iloc[profile]),
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
        if performance_table:
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
        assert ax.ax

        # Annotations
        if annotations:
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
                            (i, cast(float, table.data.iloc[profile, i]))
                        )

            for alt in range(nb_alt):
                assert performance_table
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
                            (i, cast(float, table.data.iloc[alt, i]))
                        )

        # Lexicographic order
        text = Text(
            0,
            1.2,
            "Lexicographic order : $"
            + r" \rightarrow ".join([f"P^{profile}" for profile in lexicographic_order])
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
        weights_sum = sum(self.criteria_weights.values())

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

            values = []
            # Draw the stacked barplot
            for ind_crit, crit in enumerate(performance_table.criteria):
                crit_values = [
                    self.criteria_weights[crit] / weights_sum
                    if is_better_or_equal(
                        performance_table.data.iloc[ind_alt, ind_crit],
                        self.profiles.data.iloc[profile, ind_crit],
                        cast(
                            QuantitativeScale,
                            performance_table.scales[crit],
                        ),
                    )
                    else 0
                    for profile in self.lexicographic_order
                ]
                values.append(crit_values)
            ax.add_plot(
                StackedBarPlot(
                    x,
                    values,
                    ylim=ylim,
                    xticks=xticks,
                    xticklabels=xticklabels,
                    labels=performance_table.criteria,
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
        relations = self.construct(performance_table)
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
        ranks = rankings.applymap(lambda x: value_to_rank[x])
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
        yminorticks = arange(1, nb_ranks + 2) - 0.5
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
                {k: v for k, v in zip(*unique(ranks.loc[profile], return_counts=True))}
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
            offsets = where(current_ranks == previous_ranks, offsets, 0.5).tolist()
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
