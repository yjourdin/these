from numpy.random import default_rng
from pandas import read_csv

from ..model import Model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .argument_parser import parse_args
from .generate import all_comparisons, noisy_comparisons, random_comparisons
from .io import to_csv

# Parse arguments
args = parse_args()


# Import data
model = Model.from_json(args.model.read())

A = NormalPerformanceTable(read_csv(args.A, header=None))


# Create preference structure
if args.n > 0:
    D = random_comparisons(args.n, A, model, default_rng(args.seed))
else:
    D = all_comparisons(A, model)


# Add errors
if args.error:
    D = noisy_comparisons(D, args.error, default_rng(args.seed))


# Write results
to_csv(D, args.output)
