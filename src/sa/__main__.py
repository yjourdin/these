from numpy.random import default_rng
from pandas import read_csv

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv
from .argument_parser import parse_args
from .main import learn_sa

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))

D = from_csv(args.D)


# Create random seeds
rng_init, rng_sa = (
    (default_rng(args.seed_init), default_rng(args.seed_sa))
    if (args.seed_init is not None) and (args.seed_sa is not None)
    else default_rng(args.seed).spawn(2)
)


# Learn SA
best_model, best_fitness, time, it = learn_sa(
    args.model,
    args.k,
    A,
    D,
    args.alpha,
    rng_init,
    rng_sa,
    args.T0,
    args.accept,
    args.L,
    args.amp,
    args.Tf,
    args.max_time,
    args.max_it,
    args.max_it_non_improving,
    args.log,
)


# Write results
if args.output is not None:
    args.output.write(best_model.to_json())
if args.result is not None:
    args.result.write(f"{best_fitness} {time} {it}\n")
