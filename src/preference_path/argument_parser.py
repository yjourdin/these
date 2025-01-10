import argparse

from ..constants import DEFAULT_MAX_TIME

parser = argparse.ArgumentParser()
parser.add_argument("model", type=argparse.FileType("r"), help="Collective model")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument("-o", "--output", help="Output filename")
parser.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
parser.add_argument("-s", "--seed", default=0, type=int, help="Random seed")
parser.add_argument(
    "-R", nargs="+", type=argparse.FileType("r"), help="Refused comparisons"
)


def parse_args():
    return parser.parse_args()
