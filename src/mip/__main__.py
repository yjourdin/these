import csv
from mcda.relations import PreferenceStructure
from pandas import read_csv

from ..models import GroupModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv
from .argument_parser import parse_args
from .main import learn_mip

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))

D: list[PreferenceStructure] = []
for d in args.D:
    D.append(from_csv(d))


# Learn MIP
best_model, best_fitness, time = learn_mip(
    GroupModelEnum((args.model, set(args.shared))),  # type: ignore
    args.k,
    A,
    D,
    gamma=args.gamma,
    inconsistencies=not args.no_inconsistencies,
    seed=args.seed,
    verbose=args.verbose,
)


# Write results
args.output.write(best_model.to_json() if best_model else best_model)
writer = csv.writer(args.result, "unix")
writer.writerow([best_fitness, time])
