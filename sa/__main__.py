from numpy.random import default_rng
from pandas import read_csv

from performance_table.normal_performance_table import NormalPerformanceTable
from preference_structure.io import from_csv

from .argument_parser import parse_args
from .main import learn_sa

args = parse_args()

A = NormalPerformanceTable(read_csv(args.A, header=None))

D = from_csv(args.D.read())

best_model, best_fitness, time, it = learn_sa(
    args.model,
    args.k,
    A,
    D,
    args.T0,
    args.alpha,
    args.amp,
    default_rng(args.seed_initial),
    default_rng(args.seed_sa),
    args.L,
    args.Tf,
    args.max_time,
    args.max_iter,
    args.max_iter_non_improving,
    args.log,
)

args.output.write(best_model.to_json())
args.result.write(
    f"{args.A.name},"
    f"{args.D.name},"
    f"{args.k},"
    f"{time},"
    f"{it},"
    f"{best_fitness}\n"
)
