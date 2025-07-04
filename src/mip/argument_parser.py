import argparse
from sys import stdout

from ..constants import DEFAULT_MAX_TIME, EPSILON
from ..models import ModelEnum
from ..srmp.model import SRMPParamFlag

parser = argparse.ArgumentParser()
parser.add_argument("model", type=ModelEnum, choices=ModelEnum, help="Model")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", nargs="+", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument(
    "-l", "--lex-order", nargs="+", type=int, help="Lexicographic order"
)
parser.add_argument(
    "--shared",
    nargs="+",
    default=[],
    type=SRMPParamFlag,
    choices=SRMPParamFlag,
    help="Parameters shared between decision makers",
)
parser.add_argument(
    "-c", "--collective", action="store_true", help="Elicit collective model"
)
parser.add_argument("--group", action="store_true", help="Elicit group models")
parser.add_argument(
    "--changes", nargs="+", type=int, help="Preferences previously changed"
)
parser.add_argument(
    "--refused", nargs="+", type=argparse.FileType("r"), help="Refused preferences"
)
parser.add_argument(
    "--accepted", type=argparse.FileType("r"), help="Accepted preferences"
)
parser.add_argument("--reference", type=argparse.FileType("r"), help="Reference model")
parser.add_argument("--profile-amp", type=float, help="Profiles amplitude")
parser.add_argument("--weight-amp", type=float, help="Weights amplitude")
parser.add_argument(
    "--references", nargs="+", type=argparse.FileType("r"), help="Reference models"
)
parser.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
parser.add_argument(
    "-g",
    "--gamma",
    default=EPSILON,
    type=float,
    help="Value used for modeling strict inequalities",
)
parser.add_argument(
    "-n",
    "--no-inconsistencies",
    action="store_true",
    help="Inconsistent comparisons will not be taken into account",
)
parser.add_argument(
    "-o", "--output", default=stdout, type=argparse.FileType("w"), help="Output file"
)
parser.add_argument(
    "-r", "--result", default=stdout, type=argparse.FileType("a"), help="Result file"
)
parser.add_argument("-s", "--seed", default=0, type=int, help="Random seed")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")


def parse_args():
    return parser.parse_args()
