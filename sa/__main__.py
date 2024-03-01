import argparse
import sys

from numpy.random import default_rng
from pandas import read_csv

from performance_table.core import NormalPerformanceTable
from preference_structure.core import from_csv

from .main import learn_sa

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
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Output file",
)
parser.add_argument(
    "-r",
    "--result",
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Result file",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")

args = parser.parse_args()

A = NormalPerformanceTable(read_csv(args.A))

D = from_csv(args.D.read())

SA = learn_sa(
    args.model,
    args.k,
    A,
    D,
    args.T0,
    args.alpha,
    default_rng(args.seed_initial),
    default_rng(args.seed_sa),
    args.L,
    args.Tf,
    args.max_time,
    args.max_iter,
    args.max_iter_non_improving,
    args.verbose,
)

args.output.write(SA.best_model.to_json())
args.result.write(
    f"{args.A.name},"
    f"{args.D.name},"
    f"{args.k},"
    f"{SA.time},"
    f"{SA.it},"
    f"{1-SA.best_objective}\n"
)
