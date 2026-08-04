from collections.abc import Collection
from enum import Enum, member
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.model import GroupModel, Model
from src.performance_table.type import PerformanceTableType
from src.preference_path.neighborhood import (
    Neighborhood,
    NeighborhoodCombined,
    NeighborhoodImportanceRelation,
    NeighborhoodLexOrder,
    NeighborhoodProfile,
    NeighborhoodWeight,
)
from src.preference_structure.fitness import fitness_outranking
from src.preference_structure.utils import RankingSeries
from src.rmp.model import FrozenRMPModel, RMPModel
from src.srmp.model import FrozenSRMPModel, SRMPModel

from ..preference_path.a_star import Astar
from ..random import rng_
from ..utils import kendalltau_distance


class ConsensusResult(NamedTuple):
    between_individual: Collection[Collection[float]]
    individual: Collection[float]
    among_dm: float
    between_individual_and_collective: Collection[float]
    collective: float


class DistanceRankingEnum(Enum):
    @member
    def FITNESS(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return fitness_outranking(Ra, Rb)

    @member
    def KENDALL(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return kendalltau(Ra, Rb).statistic  # pyright: ignore[reportAttributeAccessIssue]

    @member
    def SPEARMAN(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return float(spearmanr(Ra, Rb).statistic)  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]

    def __call__(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return self.value(self, Ra, Rb)

    def __str__(self) -> str:
        return self.name


def rccd(distance: DistanceRankingEnum):
    def func(ra: RankingSeries, rb: RankingSeries) -> float:
        return 0.5 * (1 + distance(ra, rb))

    return func if distance is not DistanceRankingEnum.FITNESS else distance


def distance_ranking_model(
    Ma: Model,
    Mb: Model,
    performance_table: PerformanceTableType,
    distance: DistanceRankingEnum,
):
    return distance(
        Ma.rank_series(performance_table), Mb.rank_series(performance_table)
    )


def distance_ranking_group_model(
    Ma: GroupModel[Any],
    Mb: GroupModel[Any],
    performance_table: PerformanceTableType,
    distance: DistanceRankingEnum,
):
    return [
        distance_ranking_model(Ma[dm], Mb[dm], performance_table, distance)
        for dm in range(Ma.group_size)
    ]


def consensus_group_model(
    model: GroupModel[Any],
    performance_table: PerformanceTableType,
    distance: DistanceRankingEnum,
):
    DMS = range(len(model))
    NB_DM = len(DMS)
    dm_rankings = [model[dm].rank_series(performance_table) for dm in DMS]
    collective_ranking = model.collective_model.rank_series(performance_table)

    between_individual = np.array([
        [distance(dm_rankings[dm_a], dm_rankings[dm_b]) for dm_b in DMS if dm_b != dm_a]
        for dm_a in DMS
    ])
    individual = between_individual.sum(1) / (NB_DM - 1)
    among_dm = individual.sum() / NB_DM
    between_individual_and_collective = np.array([
        distance(dm_rankings[dm_a], collective_ranking) for dm_a in DMS
    ])
    collective = between_individual_and_collective.sum() / NB_DM

    return ConsensusResult(
        between_individual,
        individual,
        among_dm,
        between_individual_and_collective,
        collective,
    )


def distance_parameter_model(
    Ma: RMPModel | SRMPModel,
    Mb: RMPModel | SRMPModel,
    performance_table: PerformanceTableType,
):
    Ma_frozen = Ma.frozen
    Mb_frozen = Mb.frozen

    neighborhoods: list[Neighborhood[FrozenRMPModel | FrozenSRMPModel]] = [
        NeighborhoodProfile(performance_table),
    ]

    if isinstance(Ma, RMPModel) and isinstance(Mb, RMPModel):
        neighborhoods.append(NeighborhoodImportanceRelation())
    if isinstance(Ma, SRMPModel) and isinstance(Mb, SRMPModel):
        neighborhoods.append(NeighborhoodWeight())

    if len(Ma.lexicographic_order) == len(Mb.lexicographic_order) > 1:
        neighborhoods.append(NeighborhoodLexOrder())

    neighborhood = NeighborhoodCombined(neighborhoods, rng_(0))

    def heuristic(model: FrozenRMPModel | FrozenSRMPModel):
        result: float = 0

        # for prof_ind, profile in enumerate(model.profiles):
        #     for crit_ind in range(len(profile)):
        #         alt = performance_table.data.to_numpy()[:, crit_ind]
        #         if (prof_a := profile[crit_ind]) < (
        #             prof_b := Mb_frozen.profiles[prof_ind][crit_ind]
        #         ):
        #             result += np.sum((prof_a < alt) & (alt < prof_b))
        #         elif prof_b < prof_a:
        #             result += np.sum((prof_b < alt) & (alt < prof_a))

        result += sum(abs(model[prof_ind][crit_ind] - Mb_frozen[prof_ind][crit_ind]) for crit_ind in range(len(model.profiles[0])) for profile_ind in range(len(model.profiles)))

        if isinstance(model, FrozenRMPModel):
            result += float(
                np.sum(
                    abs(
                        np.array(model.importance_relation)
                        - np.array(Mb_frozen.importance_relation)
                    )
                )
            )
        else:
            result += float(
                np.sum(
                    abs(
                        np.array(model.weights)
                        - np.array(Mb_frozen.weights)
                    )
                )
            )

        if len(model.lexicographic_order) > 1:
            result += kendalltau_distance(
                model.lexicographic_order, Mb_frozen.lexicographic_order
            )

        return result

    a_star = Astar(neighborhood, heuristic, latest=True)
    return len(a_star([Ma_frozen])[0]) - 1
