import argparse
from sys import stdout

from .model import SRMPParamFlag

parser = argparse.ArgumentParser()
parser.add_argument("group_size", type=int, default=1, help="Group size")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument(
    "--shared",
    nargs="*",
    default=[],
    type=SRMPParamFlag,
    choices=SRMPParamFlag,
    help="Parameters shared between decision makers",
)
parser.add_argument(
    "-p",
    "--profiles-values",
    type=argparse.FileType("w"),
    help="Possible values for profiles",
)
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-o",
    "--output",
    default=stdout,
    type=argparse.FileType("w"),
    help="Output file",
)


def parse_args():
    return parser.parse_args()
