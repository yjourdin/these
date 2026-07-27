import csv
from typing import Any, cast

from pandas import read_csv

from ..model import GroupModel
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..utils import file_or_stdout
from .args import ARGS, ArgumentsConsensus, ArgumentsDistance, TestEnum
from .main import test_consensus, test_distance

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))
distance = ARGS.distance

match ARGS.test:
    case TestEnum.DISTANCE:
        assert isinstance(ARGS, ArgumentsDistance)
        with ARGS.model_A.open() as f:
            Ma = model_from_json(f.read())
        with ARGS.model_B.open() as f:
            Mb = model_from_json(f.read())

        results = test_distance(Ma, Mb, A, distance)
    case TestEnum.CONSENSUS:
        assert isinstance(ARGS, ArgumentsConsensus)
        with ARGS.model.open() as f:
            model = cast(GroupModel[Any], model_from_json(f.read()))
        results = test_consensus(model, A, distance)


with file_or_stdout(ARGS.result, "w", "") as f:
    writer = csv.writer(f, dialect="unix")
    writer.writerows(results)
