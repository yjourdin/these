from collections.abc import Collection
from enum import Enum, member
from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt
from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from scipy.stats import kendalltau, spearmanr

from ..model import GroupModel, Model
from ..preference_structure.fitness import fitness_outranking


class ConsensusResult(NamedTuple):
    between_individual: Collection[Collection[float]]
    individual: Collection[float]
    among_dm: float
    between_individual_and_collective: Collection[float]
    collective: float


class DistanceRankingEnum(Enum):
    @member
    def FITNESS(self, Ra: Ranking, Rb: Ranking) -> float:
        return fitness_outranking(Ra, Rb)

    @member
    def KENDALL(self, Ra: Ranking, Rb: Ranking) -> float:
        return kendalltau(Ra.data, Rb.data).statistic

    @member
    def SPEARMAN(self, Ra: Ranking, Rb: Ranking) -> float:
        return spearmanr(Ra.data, Rb.data).statistic  # type: ignore

    def __call__(self, Ra: Ranking, Rb: Ranking) -> float:
        return self.value(self, Ra, Rb)

    def __str__(self) -> str:
        return self.name


def rccd(distance: DistanceRankingEnum):
    return (
        (lambda ra, rb: 0.5 * (1 + distance(ra, rb)))
        if distance is not DistanceRankingEnum.FITNESS
        else distance
    )


def distance_model(
    Ma: Model,
    Mb: Model,
    performance_table: PerformanceTable,
    distance: DistanceRankingEnum,
) -> float:
    return distance(Ma.rank(performance_table), Mb.rank(performance_table))


def distance_group_model(
    Ma: GroupModel,
    Mb: GroupModel,
    performance_table: PerformanceTable,
    distance: DistanceRankingEnum,
) -> list[float]:
    return [
        distance_model(Ma[dm], Mb[dm], performance_table, distance)
        for dm in range(Ma.group_size)
    ]


def consensus_group_model(
    model: GroupModel[Model],
    performance_table: PerformanceTable,
    distance: DistanceRankingEnum,
):
    DMS = range(len(model))
    NB_DM = len(DMS)
    dm_rankings = [model[dm].rank(performance_table) for dm in DMS]
    collective_ranking = model.collective_model.rank(performance_table)

    between_individual = cast(
        npt.NDArray[np.float64],
        np.array(
            [
                [
                    distance(dm_rankings[dm_a], dm_rankings[dm_b])
                    for dm_b in DMS
                    if dm_b != dm_a
                ]
                for dm_a in DMS
            ]
        ),
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
