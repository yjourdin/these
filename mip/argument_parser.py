import argparse
from sys import stdout

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument(
    "-g",
    "--gamma",
    default=0.001,
    type=float,
    help="Value used for modeling strict inequalities",
)
parser.add_argument(
    "-n",
    "--no-inconsistencies",
    action="store_true",
    help="inconsistent comparisons will not be taken into account",
)
parser.add_argument(
    "-o",
    "--output",
    default=stdout,
    type=argparse.FileType("w"),
    help="Output file",
)
parser.add_argument(
    "-r",
    "--result",
    default=stdout,
    type=argparse.FileType("w"),
    help="Result file",
)


def parse_args():
    return parser.parse_args()
