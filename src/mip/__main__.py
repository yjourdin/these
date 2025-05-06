import csv
from functools import reduce

from mcda.relations import PreferenceStructure
from pandas import read_csv

from ..models import GroupModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv
from ..random import rng, seed
from ..srmp.model import SRMPModel, SRMPParamFlag
from .argument_parser import parse_args
from .main import learn_mip

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))

D: list[PreferenceStructure] = []
for d in args.D:
    D.append(from_csv(d))

Refused: list[PreferenceStructure] | None = None
if args.refused:
    Refused = []
    for r in args.refused:
        Refused.append(from_csv(r))

Accepted = None
if args.accepted:
    Accepted = from_csv(args.accepted)

ref = None
if args.reference:
    ref = SRMPModel.from_json(args.reference.read())

# Create random seeds
rng_lex, rng_mip = rng(args.seed).spawn(2)


# Learn MIP
best_model, best_fitness, time = learn_mip(
    GroupModelEnum((
        args.model,
        reduce(lambda x, y: x | y, args.shared, SRMPParamFlag(0)),
    )),
    args.k,
    A,
    D,
    rng_lex,
    seed(rng_mip),
    args.max_time,
    args.lex_order,
    args.collective,
    args.changes,
    Refused,
    Accepted,
    ref,
    gamma=args.gamma,
    inconsistencies=not args.no_inconsistencies,
    verbose=args.verbose,
)


# Write results
args.output.write(best_model.to_json() if best_model else best_model)
writer = csv.writer(args.result, "unix")
writer.writerow([best_fitness, time])
