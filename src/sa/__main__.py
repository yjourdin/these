import csv

from pandas import read_csv

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv
from ..random import rng_
from .argument_parser import parse_args
from .main import learn_sa

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))

D = from_csv(args.D)


# Create random seeds
rng_init, rng_sa = (
    (args.seed_init, args.seed_sa)
    if (args.seed_init is not None) and (args.seed_sa is not None)
    else rng_(args.seed).spawn(2)
)


# Learn SA
best_model, best_fitness, time, it = learn_sa(
    args.model,
    args.k,
    A,
    D,
    args.alpha,
    args.amp,
    rng_init,
    rng_sa,
    None,
    args.T0,
    args.accept,
    args.L,
    args.Tf,
    args.max_time,
    args.max_it,
    args.max_it_non_improving,
    args.log,
)


# Write results
args.output.write(best_model.to_json() + "\n")
writer = csv.writer(args.result, "unix")
writer.writerow([best_fitness, time, it])
