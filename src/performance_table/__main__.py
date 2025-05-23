from .argument_parser import parse_args
from .normal_performance_table import NormalPerformanceTable

# Parse arguments
args = parse_args()


# Create performance table
A = NormalPerformanceTable.random(args.n, args.m, args.seed)


# Write results
A.data.to_csv(args.output, header=False, index=False)
