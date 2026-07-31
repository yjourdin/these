from typing import Any

from src.model import Group, GroupModel, Model
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.utils import Cell, add_str_to_list

from .test import (
    DistanceRankingEnum,
    consensus_group_model,
    distance_ranking_group_model,
    distance_ranking_model,
)


def test_consensus(  # pyright: ignore[reportUnknownParameterType]
    model: GroupModel[Any], A: NormalPerformanceTable, distance: DistanceRankingEnum
):
    result = consensus_group_model(model, A, distance)
    for attr, value in result._asdict().items():
        yield from add_str_to_list(value, prefix=[attr])


def test_distance_ranking(  # pyright: ignore[reportUnknownParameterType]
    Ma: Model, Mb: Model, A: NormalPerformanceTable, distance: DistanceRankingEnum
):
    match Ma, Mb:
        case GroupModel(), GroupModel():
            yield from add_str_to_list(
                distance_ranking_group_model(Ma, Mb, A, distance),  # type: ignore
                prefix=[str(distance)],
            )
        case GroupModel(), _:
            yield from add_str_to_list(
                distance_ranking_group_model(
                    Ma, Group([Mb] * Ma.group_size), A, distance
                ),  # type: ignore
                prefix=[str(distance)],
            )
        case _, GroupModel():
            yield from add_str_to_list(
                distance_ranking_group_model(
                    Group([Ma] * Mb.group_size), Mb, A, distance
                ),  # type: ignore
                prefix=[str(distance)],
            )
        case _, _:
            yield Cell(str(distance), distance_ranking_model(Ma, Mb, A, distance))
