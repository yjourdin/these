import argparse
from typing import cast

from .arguments import ExperimentEnum
from .experiments.elicitation.arguments import ArgumentsElicitation
from .experiments.group_decision.arguments import ArgumentsGroupDecision

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "experiment", type=ExperimentEnum, choices=ExperimentEnum, help="Experiment"
)
parser.add_argument("args", type=argparse.FileType("r"), help="Arguments file")
parser.add_argument("-d", "--dir", type=str, help="Results directory")
parser.add_argument("-n", "--name", type=str, help="Experiment name")
parser.add_argument("-j", "--jobs", type=int, help="Number of jobs")
parser.add_argument("-s", "--stop-error", action="store_true", help="Stop on error")
parser.add_argument(
    "-e", "--extend", action="store_true", help="Extend previous experiment"
)


args = parser.parse_args()

with args.args as file:
    match cast(ExperimentEnum, args.experiment):
        case ExperimentEnum.ELICITATION:
            ARGS = ArgumentsElicitation.from_json(file.read())
        case ExperimentEnum.GROUP_DECISION:
            ARGS = ArgumentsGroupDecision.from_json(file.read())  # pyright: ignore[reportConstantRedefinition]

ARGS.dir = args.dir or ARGS.dir
ARGS.name = args.name or ARGS.name
ARGS.jobs = args.jobs or ARGS.jobs
ARGS.stop_error = args.stop_error
ARGS.extend = args.extend
