from mcda.relations import PreferenceStructure
from pandas import read_csv

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv
from .argument_parser import parse_args
from .main import learn_mip

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))

D = PreferenceStructure()
for d in args.D:
    D._relations += from_csv(d)._relations


# Learn MIP
best_model, best_fitness, time = learn_mip(
    args.k, A, D, args.gamma, not args.no_inconsistencies, args.seed, args.verbose
)


# Write results
if args.output is not None:
    args.output.write(best_model.to_json() if best_model is not None else "None")
if args.result is not None:
    args.result.write(f"{best_fitness} {best_fitness}\n")
