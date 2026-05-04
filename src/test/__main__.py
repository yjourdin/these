import csv

from pandas import read_csv

from ..model import GroupModel
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .args import ARGS, TestEnum
from .main import test_consensus, test_distance

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))
distance = ARGS.distance
writer = csv.writer(ARGS.result)

match ARGS.test:
    case TestEnum.DISTANCE:
        Ma = model_from_json(ARGS.model_A.read())
        Mb = model_from_json(ARGS.model_B.read())

        writer.writerows(test_distance(Ma, Mb, A, distance))
    case TestEnum.CONSENSUS:
        model = model_from_json(ARGS.model.read())
        assert isinstance(model, GroupModel)
        writer.writerows(test_consensus(model, A, distance))  # type: ignore
    case _:
        raise ValueError(f"Unknown test {ARGS.test}")
