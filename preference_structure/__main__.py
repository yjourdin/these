from numpy.random import default_rng
from pandas import read_csv

from performance_table.core import NormalPerformanceTable
from rmp.model import RMPModel
from srmp.model import SRMPModel

from .argument_parser import parse_args
from .core import to_csv
from .generate import all_comparisons, noisy_comparisons, random_comparisons

args = parse_args()

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
