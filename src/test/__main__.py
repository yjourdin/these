import csv
from typing import Any, cast

from pandas import read_csv

from src.model import GroupModel
from src.models import model_from_json
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.utils import Cell, file_or_stdout

from ..rmp.model import RMPModel
from ..srmp.model import SRMPModel
from .args import (
    ARGS,
    ArgumentsConsensus,
    ArgumentsDistance,
    ArgumentsParameters,
    ArgumentsRanking,
    RankingEnum,
    TestEnum,
)
from .main import test_consensus, test_distance_ranking
from .test import distance_parameter_model

# Import data
A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

match ARGS.test:
    case TestEnum.PARAMETERS:
        ARGS = cast(ArgumentsParameters, ARGS)
        with ARGS.model_A.open() as f:
            Ma = model_from_json(f.read())
        with ARGS.model_B.open() as f:
            Mb = model_from_json(f.read())

        if not isinstance(Ma, (RMPModel, SRMPModel)) or not isinstance(
            Mb, (RMPModel, SRMPModel)
        ):
            raise TypeError("Random model is not accepted")
        results = (Cell("Changes", distance_parameter_model(Ma, Mb, A)),)
    case TestEnum.RANKING:
        ARGS = cast(ArgumentsRanking, ARGS)
        distance = ARGS.distance
        match ARGS.ranking:
            case RankingEnum.DISTANCE:
                ARGS = cast(ArgumentsDistance, ARGS)
                with ARGS.model_A.open() as f:
                    Ma = model_from_json(f.read())
                with ARGS.model_B.open() as f:
                    Mb = model_from_json(f.read())

                results = test_distance_ranking(Ma, Mb, A, distance)
            case RankingEnum.CONSENSUS:
                ARGS = cast(ArgumentsConsensus, ARGS)
                with ARGS.model.open() as f:
                    model = cast(GroupModel[Any], model_from_json(f.read()))
                results = test_consensus(model, A, distance)


with file_or_stdout(ARGS.result, "w", "") as f:
    writer = csv.writer(f, dialect="unix")
    writer.writerows(results)  # pyright: ignore[reportUnknownArgumentType]
