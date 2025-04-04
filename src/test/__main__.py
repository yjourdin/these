import csv

from pandas import read_csv

from ..model import GroupModel
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .argument_parser import TestEnum, parse_args
from .main import test_consensus, test_distance

# Parse arguments
args = parse_args()


# Import data
A = NormalPerformanceTable(read_csv(args.A, header=None))
distance = args.distance
writer = csv.writer(args.result)

match args.test:
    case TestEnum.DISTANCE:
        Ma = model_from_json(args.model_A.read())
        Mb = model_from_json(args.model_B.read())

        writer.writerows(test_distance(Ma, Mb, A, distance))
    case TestEnum.CONSENSUS:
        model = model_from_json(args.model.read())
        assert isinstance(model, GroupModel)
        writer.writerows(test_consensus(model, A, distance))  # type: ignore
    case _:
        raise ValueError(f"Unknown test {args.test}")
