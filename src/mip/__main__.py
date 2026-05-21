import csv
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from math import inf
from operator import attrgetter

from mcda.relations import PreferenceStructure
from pandas import read_csv

from src.models import GroupModelEnum
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv
from src.random import seed_
from src.srmp.model import SRMPModel, SRMPParamFlag

from ..utils import catchtime, file_or_stdout
from .args import ARGS
from .main import SenseEnum, create_mip, mip_result

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

D: list[PreferenceStructure] = []
for d in ARGS.D:
    with d.open("r") as f:
        D.append(from_csv(f))

Refused = None
if ARGS.refused:
    with ARGS.refused.open("r", newline="") as f:
        Refused = from_csv(f)

Accepted = None
if ARGS.accepted:
    with ARGS.accepted.open("r", newline="") as f:
        Accepted = from_csv(f)

refs: list[SRMPModel] | None = None
if ARGS.references:
    refs = []
    for ref in ARGS.references:
        with ref.open("r") as f:
            refs.append(SRMPModel.from_json(f.read()))

ref = None
if ARGS.reference:
    with ARGS.reference.open("r") as f:
        ref = SRMPModel.from_json(f.read())

# Create random seeds
seed_lex, seed_mip = seed_(ARGS.seed).spawn(2)


# Generate MIP
mips, sense = create_mip(
    GroupModelEnum((
        ARGS.model,
        reduce(lambda x, y: x | y, ARGS.shared, SRMPParamFlag.NONE),
    )),
    ARGS.k,
    A,
    D,
    seed_lex,
    seed_mip,
    ARGS.max_time,
    ARGS.lex_order,
    ARGS.collective,
    ARGS.group,
    ARGS.changes,
    Refused,
    Accepted,
    reference_model=ref,
    profiles_amp=ARGS.profile_amp,
    weights_amp=ARGS.weight_amp,
    reference_models=refs,
    gamma=ARGS.gamma,
    inconsistencies=not ARGS.no_inconsistencies,
    verbose=ARGS.verbose,
    log_path=ARGS.log_path,
    nb_cpus=ARGS.nb_cpus,
)

with catchtime() as time, ThreadPoolExecutor(ARGS.nb_cpus) as thread_pool:
    results = list(thread_pool.map(mip_result, mips))

for i, result in enumerate(results):
    if result.best_model is None:
        match sense:
            case SenseEnum.MIN:
                placeholder = inf
            case SenseEnum.MAX:
                placeholder = -inf
        results[i] = result._replace(best_objective=placeholder)
optimal = all(result.optimal for result in results)
best_model, best_objective, _, _ = sense.value(
    results, key=attrgetter("best_objective"), default=(None,) * 4
)


# Write output
with file_or_stdout(ARGS.output, "w") as f:
    f.write(best_model.to_json() if best_model else "")

# Write results
with file_or_stdout(ARGS.result, "w", "") as f:
    writer = csv.writer(f, "unix")
    writer.writerow([
        best_objective if best_objective != placeholder else None,
        time(),
        optimal,
    ])  # pyright: ignore[reportPossiblyUnboundVariable]
