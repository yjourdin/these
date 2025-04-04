import argparse
from sys import stdout

from ..constants import DEFAULT_MAX_TIME
from ..models import ModelEnum

parser = argparse.ArgumentParser()


parser.add_argument("model", type=ModelEnum, choices=ModelEnum, help="Model")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", type=argparse.FileType("r"), help="Comparisons")

init_group = parser.add_mutually_exclusive_group(required=True)
init_group.add_argument("--T0", type=float, help="Initial temperature")
init_group.add_argument(
    "--accept", type=float, help="Acceptance rate to determine initial temperature"
)

parser.add_argument("--alpha", type=float, required=True, help="Cooling coefficient")
parser.add_argument(
    "--amp", type=float, help="Weight neighborhood amplitude", default=2
)
parser.add_argument("--L", default=1, type=int, help="Length of Markov chains")

stop_group = parser.add_mutually_exclusive_group(required=True)
stop_group.add_argument("--Tf", type=float, help="Final temperature")
stop_group.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
stop_group.add_argument("--max-it", type=int, help="Max number of iterations")
stop_group.add_argument(
    "--max-it-non-improving", type=int, help="Max number of non improving iterations"
)

parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("--seed-init", type=int, help="Initial model random seed")
parser.add_argument("--seed-sa", type=int, help="Simulated annealing random seed")
parser.add_argument(
    "-o", "--output", default=stdout, type=argparse.FileType("w"), help="Output file"
)
parser.add_argument(
    "-r", "--result", default=stdout, type=argparse.FileType("a"), help="Result file"
)
parser.add_argument(
    "-l", "--log", nargs="?", type=argparse.FileType("w"), const=stdout, help="Log file"
)


def parse_args():
    return parser.parse_args()
