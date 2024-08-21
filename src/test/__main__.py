from pandas import read_csv

from src.model import GroupModel

from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .argument_parser import parse_args
from .main import test

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))
Mo = model_from_json(args.Mo.read())
Me = model_from_json(args.Me.read())

NB_DM = Mo.size if isinstance(Mo, GroupModel) else 1


# Test
fitness, kendall_tau = test(A, Mo, Me)


# Write results
if args.result is not None:
    args.result.write(f"{fitness} {kendall_tau}\n")
