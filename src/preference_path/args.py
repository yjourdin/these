import argparse
from sys import stdout

from src.constants import DEFAULT_MAX_TIME

parser = argparse.ArgumentParser()
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument("models", nargs="+", type=argparse.FileType("r"), help="Collective models")
parser.add_argument("-o", "--output", help="Output filename")
parser.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-R", type=argparse.FileType("r"), help="Refused comparisons"
)
parser.add_argument("--model-output", help="Output model files")
parser.add_argument(
    "-r", "--result", default=stdout, type=argparse.FileType("a"), help="Result file"
)


ARGS = parser.parse_args()
