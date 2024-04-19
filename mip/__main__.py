from mcda.relations import PreferenceStructure
from pandas import read_csv

from performance_table.normal_performance_table import NormalPerformanceTable
from preference_structure.io import from_csv

from .argument_parser import parse_args
from .main import learn_mip

args = parse_args()

A = NormalPerformanceTable(read_csv(args.A, header=None))

D = PreferenceStructure()
for d in args.D:
    D._relations += from_csv(d.read())._relations

best_model, best_fitness, time = learn_mip(
    args.k, A, D, args.gamma, not args.no_inconsistencies, args.seed, args.verbose
)

if best_model is not None:
    args.output.write(best_model.to_json())
    args.result.write(
        f"{args.A.name}," f"{args.D.name}," f"{args.k}," f"{time}," f"{best_fitness}\n"
    )
else:
    args.output.write("None")
    args.result.write(
        f"{args.A.name}," f"{args.D.name}," f"{args.k}," f"{time}," f"{best_fitness}\n"
    )
