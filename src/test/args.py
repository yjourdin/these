import argparse
from dataclasses import dataclass
from pathlib import Path

from ..case_insensitive_str_enum import CaseInsensitiveStrEnum
from ..dataclass import Dataclass
from .test import DistanceRankingEnum


class TestEnum(CaseInsensitiveStrEnum):
    DISTANCE = "D"
    CONSENSUS = "C"


parser = argparse.ArgumentParser()
parser.add_argument("A", type=Path, help="Alternatives")
parser.add_argument(
    "distance",
    type=DistanceRankingEnum.__getitem__,
    choices=DistanceRankingEnum,
    help="Distance to use",
)

subparsers = parser.add_subparsers(dest="test", required=True, help="Test to compute")

parser_distance = subparsers.add_parser(TestEnum.DISTANCE, help="Distance test")
parser_distance.add_argument("model_A", type=Path, help="First model")
parser_distance.add_argument("model_B", type=Path, help="Second model")

parser_consensus = subparsers.add_parser(TestEnum.CONSENSUS, help="Consensus test")
parser_consensus.add_argument("model", nargs="+", type=Path, help="Group model")

parser.add_argument("-r", "--result", type=Path, help="Result file")


@dataclass(init=False)
class Arguments(Dataclass):
    A: Path
    distance: DistanceRankingEnum
    test: TestEnum
    model_A: Path | None = None
    model_B: Path | None = None
    model: list[Path] | None = None
    result: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
