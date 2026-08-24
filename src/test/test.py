from collections.abc import Collection
from enum import Enum, member
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from src.model import GroupModel, Model
from src.performance_table.type import PerformanceTableType
from src.preference_path.neighborhood import (
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

    def __call__(self, Ra: RankingSeries, Rb: RankingSeries):
        return self.value(self, Ra, Rb)

    def __str__(self):
        return self.name


def rccd(distance: DistanceRankingEnum):
    def func(ra: RankingSeries, rb: RankingSeries):
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

    def heuristic_profile(model: FrozenRMPModel | FrozenSRMPModel):
        # result += sum(abs(model.profiles[prof_ind][crit_ind] - Mb_frozen.profiles[prof_ind][crit_ind]) for crit_ind in range(len(model.profiles[0])) for prof_ind in range(len(model.profiles)))
        result = 0
        for prof_ind, profile in enumerate(model.profiles):
            for crit_ind in range(len(profile)):
                alt = performance_table.data.to_numpy()[:, crit_ind]
                if (prof_a := profile[crit_ind]) < (
                    prof_b := Mb_frozen.profiles[prof_ind][crit_ind]
                ):
                    result += np.sum((prof_a < alt) & (alt < prof_b))
                elif prof_b < prof_a:
                    result += np.sum((prof_b < alt) & (alt < prof_a))
        return result

    def heuristic_importance_relation(model: FrozenRMPModel | FrozenSRMPModel):
        if isinstance(model, FrozenRMPModel):
            Ia_dict = dict(model.importance_relation)
            Ib_dict = dict(Mb_frozen.importance_relation)
            keys = Ia_dict.keys() & Ib_dict.keys()
            Ia = rankdata([Ia_dict[k] for k in keys])
            Ib = rankdata([Ib_dict[k] for k in keys])
        else:
            Ia = rankdata(model.importance_relation)
            Ib = rankdata(Mb_frozen.importance_relation)
        return float(np.sum(abs(Ia - Ib)))  # pyright: ignore[reportUnknownArgumentType]

    def heuristic_lexicographic_order(model: FrozenRMPModel | FrozenSRMPModel):
        return (
            kendalltau_distance(
                model.lexicographic_order, Mb_frozen.lexicographic_order
            )
            if len(model.lexicographic_order) > 1
            else 0
        )

    a_star_profile = Astar(
        NeighborhoodProfile[FrozenRMPModel | FrozenSRMPModel](performance_table),
        heuristic_profile,
        latest=True,
    )

    if isinstance(Ma, RMPModel) and isinstance(Mb, RMPModel):
        neighborhood = NeighborhoodImportanceRelation()
    elif isinstance(Ma, SRMPModel) and isinstance(Mb, SRMPModel):
        neighborhood = NeighborhoodWeight()
    else:
        raise TypeError("different models")
    a_star_importance_relation = Astar(
        neighborhood, heuristic_importance_relation, latest=True
    )

    a_star_lexicographic_order = Astar(
        NeighborhoodLexOrder[FrozenRMPModel | FrozenSRMPModel](),
        heuristic_lexicographic_order,
        latest=True,
    )

    result = 0

    path_profile = a_star_profile([Ma_frozen])[0]
    if a_star_profile.time >= a_star_profile.max_time:
        return -1
    result += len(path_profile) - 1
    M_profile = path_profile[0]

    path_importance_relation = a_star_importance_relation([M_profile])[0]
    if a_star_importance_relation.time >= a_star_importance_relation.max_time:
        return -1
    result += len(path_importance_relation) - 1
    M_importance_relation = path_importance_relation[0]

    path_lexicographic_order = a_star_lexicographic_order([M_importance_relation])[0]
    if a_star_lexicographic_order.time >= a_star_lexicographic_order.max_time:
        return -1
    result += len(path_lexicographic_order) - 1

    return result
