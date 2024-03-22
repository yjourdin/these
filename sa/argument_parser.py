import argparse
from sys import stdout

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["RMP", "SRMP"], help="Model")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument("--T0", type=float, required=True, help="Initial temperature")
parser.add_argument("--alpha", type=float, required=True, help="Cooling coefficient")
parser.add_argument(
    "--L",
    default=1,
    type=int,
    help="Length of Markov chains",
)
stopping_criterion_group = parser.add_mutually_exclusive_group(required=True)
stopping_criterion_group.add_argument("--Tf", type=float, help="Final temperature")
stopping_criterion_group.add_argument(
    "--max-time", type=int, help="Time limit (in seconds)"
)
stopping_criterion_group.add_argument(
    "--max-iter", type=int, help="Max number of iterations"
)
stopping_criterion_group.add_argument(
    "--max-iter-non-improving",
    type=int,
    help="Max number of non improving iterations",
)
parser.add_argument("--seed-initial", type=int, help="Initial model random seed")
parser.add_argument("--seed-sa", type=int, help="Simulated annealing random seed")
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
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")


def parse_args():
    return parser.parse_args()
