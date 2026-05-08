import csv
from functools import reduce

from mcda.relations import PreferenceStructure
from pandas import read_csv

from src.models import GroupModelEnum
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv
from src.random import seed_
from src.srmp.model import SRMPModel, SRMPParamFlag

from .args import ARGS
from .main import learn_mip

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

D: list[PreferenceStructure] = []
for d in ARGS.D:
    D.append(from_csv(d))

Refused: list[PreferenceStructure] | None = None
if ARGS.refused:
    Refused = []
    for r in ARGS.refused:
        Refused.append(from_csv(r))

Accepted = None
if ARGS.accepted:
    Accepted = from_csv(ARGS.accepted)

refs: list[SRMPModel] | None = None
if ARGS.references:
    refs = []
    for ref in ARGS.references:
        refs.append(SRMPModel.from_json(ref.read()))

ref = None
if ARGS.reference:
    ref = SRMPModel.from_json(ARGS.reference.read())

# Create random seeds
seed_lex, seed_mip = seed_(ARGS.seed).spawn(2)


# Generate MIP
best_model, best_objective, time = learn_mip(
    GroupModelEnum((
        ARGS.model,
        reduce(lambda x, y: x | y, ARGS.shared, SRMPParamFlag(0)),
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
    nb_cpus=ARGS.nb_cpus,
)


# Write results
ARGS.output.write(best_model.to_json() if best_model else "")
writer = csv.writer(ARGS.result, "unix")
writer.writerow([best_objective, time])
