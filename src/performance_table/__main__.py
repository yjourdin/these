from .args import ARGS
from .normal_performance_table import NormalPerformanceTable

# Create performance table
A = NormalPerformanceTable.random(ARGS.n, ARGS.m, ARGS.seed)


# Write results
A.data.to_csv(ARGS.output, header=False, index=False)
