from numpy.random import default_rng
from pandas import read_csv

from performance_table.core import NormalPerformanceTable
from preference_structure.core import from_csv

from .argument_parser import parse_args
from .main import learn_sa

args = parse_args()

A = NormalPerformanceTable(read_csv(args.A, header=None))

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