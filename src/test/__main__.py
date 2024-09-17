import csv

from pandas import read_csv

from src.utils import add_str_to_list

from ..model import Group, GroupModel, Model
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .argument_parser import TestEnum, parse_args
from .test import consensus_group_model, distance_group_model, distance_model

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

        match Ma, Mb:
            case GroupModel(), GroupModel():
                writer.writerows(
                    add_str_to_list(
                        distance_group_model(Ma, Mb, A, distance),
                        prefix=[str(distance)],
                    )
                )
            case GroupModel(), Model():
                writer.writerows(
                    add_str_to_list(
                        distance_group_model(
                            Ma, Group([Mb] * Ma.group_size), A, distance
                        ),
                        prefix=[str(distance)],
                    )
                )
            case Model(), GroupModel():
                writer.writerows(
                    add_str_to_list(
                        distance_group_model(
                            Group([Ma] * Mb.group_size), Mb, A, distance
                        ),
                        prefix=[str(distance)],
                    )
                )
            case Model(), Model():
                writer.writerow([distance, distance_model(Ma, Mb, A, distance)])
    case TestEnum.CONSENSUS:
        model = model_from_json(args.model.read())
        assert isinstance(model, GroupModel)
        result = consensus_group_model(model, A, distance)
        for attr, value in result._asdict().items():
            for name, val in add_str_to_list(value, prefix=[attr]):
                writer.writerow([name, val])
