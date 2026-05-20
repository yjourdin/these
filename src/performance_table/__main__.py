from ..utils import file_or_stdout
from .args import ARGS
from .normal_performance_table import NormalPerformanceTable

# Create performance table
A = NormalPerformanceTable.random(ARGS.n, ARGS.m, ARGS.seed)


# Write output
with file_or_stdout(ARGS.output, "w") as f:
    A.data.to_csv(f, header=False, index=False)
