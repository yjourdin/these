from .argument_parser import parse_args
from .normal_performance_table import NormalPerformanceTable
from ..random import rng

# Parse arguments
args = parse_args()


# Create performance table
A = NormalPerformanceTable.random(args.n, args.m, rng(args.seed))


# Write results
A.data.to_csv(args.output, header=False, index=False)
