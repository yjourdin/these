from collections.abc import Collection
from enum import Enum, member
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau, spearmanr

from ..model import GroupModel, Model
from ..performance_table.type import PerformanceTableType
from ..preference_structure.fitness import fitness_outranking
from ..preference_structure.utils import RankingSeries


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
        return kendalltau(Ra, Rb).statistic

    @member
    def SPEARMAN(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return cast(float, spearmanr(Ra, Rb).statistic)

    def __call__(self, Ra: RankingSeries, Rb: RankingSeries) -> float:
        return self.value(self, Ra, Rb)

    def __str__(self) -> str:
        return self.name


def rccd(distance: DistanceRankingEnum):
    def func(ra: RankingSeries, rb: RankingSeries) -> float:
        return 0.5 * (1 + distance(ra, rb))

    return func if distance is not DistanceRankingEnum.FITNESS else distance


def distance_model(
    Ma: Model,
    Mb: Model,
    performance_table: PerformanceTableType,
    distance: DistanceRankingEnum,
) -> float:
    return distance(
        Ma.rank_series(performance_table), Mb.rank_series(performance_table)
    )


def distance_group_model(
    Ma: GroupModel[Any],
    Mb: GroupModel[Any],
    performance_table: PerformanceTableType,
    distance: DistanceRankingEnum,
) -> list[float]:
    return [
        distance_model(Ma[dm], Mb[dm], performance_table, distance)
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

    between_individual = cast(
        npt.NDArray[np.float64],
        np.array([
            [
                distance(dm_rankings[dm_a], dm_rankings[dm_b])
                for dm_b in DMS
                if dm_b != dm_a
            ]
            for dm_a in DMS
        ]),
    )
    individual = cast(
        npt.NDArray[np.float64], (between_individual.sum(1) / (NB_DM - 1))
    )
    among_dm = cast(float, individual.sum() / NB_DM)
    between_individual_and_collective = cast(
        npt.NDArray[np.float64],
        np.array([distance(dm_rankings[dm_a], collective_ranking) for dm_a in DMS]),
    )
    collective = between_individual_and_collective.sum() / NB_DM

    return ConsensusResult(
        between_individual,
        individual,
        among_dm,
        between_individual_and_collective,
        collective,
    )
