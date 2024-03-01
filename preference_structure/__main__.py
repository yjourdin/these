import argparse
import sys

from numpy.random import default_rng
from pandas import read_csv

from performance_table.core import NormalPerformanceTable
from rmp.model import RMPModel
from srmp.model import SRMPModel

from .core import to_csv
from .generate import all_comparisons, noisy_comparisons, random_comparisons

parser = argparse.ArgumentParser()
parser.add_argument("model", type=argparse.FileType("r"), help="Preferences model")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("n", type=int, help="Number of comparisons")
parser.add_argument("-e", "--error", type=float, help="Error rate")
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-o",
    "--output",
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Output file",
)

args = parser.parse_args()

s = args.model.read()
if "capacities" in s:
    model = RMPModel.from_json(s)
elif "weights" in s:
    model = SRMPModel.from_json(s)
else:
    ValueError("model is not a valid model")

A = NormalPerformanceTable(read_csv(args.A))

if args.n > 0:
    D = random_comparisons(args.n, A, model, default_rng(args.seed))
else:
    D = all_comparisons(A, model)

if args.error:
    D = noisy_comparisons(D, args.error, default_rng(args.seed))

args.output.write(to_csv(D))
