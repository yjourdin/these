import csv
from concurrent.futures import ProcessPoolExecutor
from operator import attrgetter

from mcda.relations import PreferenceStructure
from pandas import read_csv

from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv
from src.random import rng_

from ..utils import catchtime, file_or_stdout
from .args import ARGS
from .main import create_sa, sa_result

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

D: list[PreferenceStructure] = []
for d in ARGS.D:
    with d.open("r") as f:
        D.append(from_csv(f))


# Create random seeds
rng_init, rng_sa = (
    (ARGS.seed_init, ARGS.seed_sa)
    if (ARGS.seed_init is not None) and (ARGS.seed_sa is not None)
    else rng_(ARGS.seed).spawn(2)
)

Refused = None
if ARGS.refused:
    with ARGS.refused.open("r", newline="") as f:
        Refused = from_csv(f)

Accepted = None
if ARGS.accepted:
    with ARGS.accepted.open("r", newline="") as f:
        Accepted = from_csv(f)

# Learn SA
sas, sense = create_sa(
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
    ARGS.verbose,
    ARGS.log_path,
    ARGS.changes,
    Accepted,
    Refused,
    rng_init,
    rng_sa,
    ARGS.nb_cpus,
)

with catchtime() as time, ProcessPoolExecutor(ARGS.nb_cpus) as process_pool:
    results = list(process_pool.map(sa_result, sas))

best_model, best_objective, _, it = sense.value(
    results, key=attrgetter("best_objective")
)

# Write output
with file_or_stdout(ARGS.output, "w") as f:
    f.write(best_model.to_json() if best_model else "")

# Write results
with file_or_stdout(ARGS.result, "w", "") as f:
    writer = csv.writer(f, "unix")
    writer.writerow([1 - best_objective, time(), it])
