import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from operator import attrgetter

from mcda.relations import PreferenceStructure
from pandas import read_csv

from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv
from src.random import rng_

from .args import ARGS
from .main import learn_sa

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

D: list[PreferenceStructure] = []
for d in ARGS.D:
    D.append(from_csv(d))


# Create random seeds
rng_init, rng_sa = (
    (ARGS.seed_init, ARGS.seed_sa)
    if (ARGS.seed_init is not None) and (ARGS.seed_sa is not None)
    else rng_(ARGS.seed).spawn(2)
)

Refused: list[PreferenceStructure] | None = None
if ARGS.refused:
    Refused = []
    for r in ARGS.refused:
        Refused.append(from_csv(r))

# Learn SA
with ProcessPoolExecutor(ARGS.nb_cpus) as process_pool:
    fn = partial(
        learn_sa,
        ARGS.model,
        ARGS.k,
        A,
        D,
        ARGS.alpha,
        ARGS.amp,
        None,
        ARGS.T0,
        ARGS.accept,
        ARGS.L,
        ARGS.Tf,
        ARGS.max_time,
        ARGS.max_it,
        ARGS.max_it_non_improving,
        ARGS.log,
        ARGS.changes,
        Refused,
    )
    best_model, best_objective, time, it = min(
        process_pool.map(
            fn, zip(rng_init.spawn(ARGS.nb_cpus), rng_sa.spawn(ARGS.nb_cpus))
        ),
        key=attrgetter("best_objective"),
    )

# Write results
ARGS.output.write(best_model.to_json() + "\n")
writer = csv.writer(ARGS.result, "unix")
writer.writerow([1 - best_objective, time, it])
