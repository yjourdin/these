import argparse
from dataclasses import replace
from pathlib import Path

from .arguments import ExperimentEnum
from .experiments.elicitation.arguments import ArgumentsElicitation
from .experiments.group_decision.arguments import ArgumentsGroupDecision

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "experiment", type=ExperimentEnum, choices=ExperimentEnum, help="Experiment"
)
parser.add_argument("args", type=Path, help="Arguments file")
parser.add_argument("-d", "--dir", type=str, help="Results directory")
parser.add_argument("-n", "--name", type=str, help="Experiment name")
parser.add_argument("-j", "--jobs", type=int, help="Number of jobs")
parser.add_argument("-s", "--stop-error", action="store_true", help="Stop on error")
parser.add_argument(
    "-e", "--extend", action="store_true", help="Extend previous experiment"
)


args = vars(parser.parse_args())

args_file: Path = args.pop("args")
experiment: ExperimentEnum = args.pop("experiment")

with args_file.open("r") as file:
    match experiment:
        case ExperimentEnum.ELICITATION:
            ARGS = ArgumentsElicitation.from_json(file.read())
        case ExperimentEnum.GROUP_DECISION:
            ARGS = ArgumentsGroupDecision.from_json(file.read())  # pyright: ignore[reportConstantRedefinition]

replace(ARGS, **args)
