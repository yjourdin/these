import argparse
from sys import stdout

from ..enum import StrEnumCustom
from .test import DistanceRankingEnum


class TestEnum(StrEnumCustom):
    DISTANCE = "D"
    CONSENSUS = "C"


parser = argparse.ArgumentParser()
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument(
    "distance",
    type=DistanceRankingEnum.__getitem__,
    choices=DistanceRankingEnum,
    help="Distance to use",
)

subparsers = parser.add_subparsers(dest="test", required=True, help="Test to compute")

parser_distance = subparsers.add_parser(TestEnum.DISTANCE, help="Distance test")
parser_distance.add_argument("model_A", type=argparse.FileType("r"), help="First model")
parser_distance.add_argument(
    "model_B", type=argparse.FileType("r"), help="Second model"
)

parser_consensus = subparsers.add_parser(TestEnum.CONSENSUS, help="Consensus test")
parser_consensus.add_argument(
    "model", nargs="+", type=argparse.FileType("r"), help="Group model"
)

parser.add_argument(
    "-r", "--result", default=stdout, type=argparse.FileType("a"), help="Result file"
)


def parse_args():
    return parser.parse_args()
