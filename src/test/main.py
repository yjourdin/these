from ..model import Group, GroupModel, Model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..utils import add_str_to_list
from .test import (
    DistanceRankingEnum,
    consensus_group_model,
    distance_group_model,
    distance_model,
)


def test_consensus(
    model: GroupModel, A: NormalPerformanceTable, distance: DistanceRankingEnum
):
    result = consensus_group_model(model, A, distance)
    for attr, value in result._asdict().items():
        yield from add_str_to_list(value, prefix=[attr])


def test_distance(
    Ma: Model, Mb: Model, A: NormalPerformanceTable, distance: DistanceRankingEnum
):
    match Ma, Mb:
        case GroupModel(), GroupModel():
            yield from add_str_to_list(
                distance_group_model(Ma, Mb, A, distance), prefix=[str(distance)]
            )
        case GroupModel(), _:
            yield from add_str_to_list(
                distance_group_model(Ma, Group([Mb] * Ma.group_size), A, distance),
                prefix=[str(distance)],
            )
        case _, GroupModel():
            yield from add_str_to_list(
                distance_group_model(Group([Ma] * Mb.group_size), Mb, A, distance),
                prefix=[str(distance)],
            )
        case _, _:
            yield str(distance), distance_model(Ma, Mb, A, distance)
