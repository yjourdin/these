import argparse
from dataclasses import dataclass
from pathlib import Path

from src.case_insensitive_str_enum import CaseInsensitiveStrEnum
from src.dataclass import Dataclass

from .test import DistanceRankingEnum


class TestEnum(CaseInsensitiveStrEnum):
    PARAMETERS = "P"
    RANKING = "R"


parser = argparse.ArgumentParser()
parser.add_argument("A", type=Path, help="Alternatives")

subparsers = parser.add_subparsers(dest="test", required=True, help="Test to compute")

parser_parameters = subparsers.add_parser(TestEnum.PARAMETERS, help="Parameters test")
parser_parameters.add_argument("model_A", type=Path, help="First model")
parser_parameters.add_argument("model_B", type=Path, help="Second model")


class RankingEnum(CaseInsensitiveStrEnum):
    DISTANCE = "D"
    CONSENSUS = "C"


parser_ranking = subparsers.add_parser(TestEnum.RANKING, help="Ranking test")
parser_ranking.add_argument(
    "distance",
    type=DistanceRankingEnum.__getitem__,
    choices=DistanceRankingEnum,
    help="Distance to use",
)

subparsers_ranking = parser_ranking.add_subparsers(
    dest="ranking", required=True, help="Ranking test to compute"
)

parser_distance = subparsers_ranking.add_parser(RankingEnum.DISTANCE, help="Distance test")
parser_distance.add_argument("model_A", type=Path, help="First model")
parser_distance.add_argument("model_B", type=Path, help="Second model")

parser_consensus = subparsers_ranking.add_parser(RankingEnum.CONSENSUS, help="Consensus test")
parser_consensus.add_argument("model", type=Path, help="Group model")

parser.add_argument("-r", "--result", type=Path, help="Result file")


@dataclass(init=False)
class Arguments(Dataclass):
    A: Path
    test: TestEnum
    result: Path | None = None


class ArgumentsParameters(Arguments):
    model_A: Path
    model_B: Path


class ArgumentsRanking(Arguments):
    ranking: RankingEnum
    distance: DistanceRankingEnum


class ArgumentsDistance(ArgumentsRanking):
    model_A: Path
    model_B: Path


class ArgumentsConsensus(ArgumentsRanking):
    model: Path


ARGS = parser.parse_args(namespace=Arguments())
