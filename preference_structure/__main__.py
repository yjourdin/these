from numpy.random import default_rng
from pandas import read_csv

from model import import_model
from performance_table.normal_performance_table import NormalPerformanceTable

from .argument_parser import parse_args
from .io import to_csv
from .generate import all_comparisons, noisy_comparisons, random_comparisons

args = parse_args()

s = args.model.read()
model = import_model(s)

A = NormalPerformanceTable(read_csv(args.A))

if args.n > 0:
    D = random_comparisons(args.n, A, model, default_rng(args.seed))
else:
    D = all_comparisons(A, model)

if args.error:
    D = noisy_comparisons(D, args.error, default_rng(args.seed))

args.output.write(to_csv(D))
