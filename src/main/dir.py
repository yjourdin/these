from .args import ARGS
from .arguments import ExperimentEnum
from .directory import Directory
from .experiments.elicitation.directory import DirectoryElicitation
from .experiments.group_decision.directory import DirectoryGroupDecision

directory_class = Directory
match ARGS.experiment:
    case ExperimentEnum.ELICITATION:
        directory_class = DirectoryElicitation
    case ExperimentEnum.GROUP_DECISION:
        directory_class = DirectoryGroupDecision


# Initialise directory
DIR = directory_class(ARGS.name, ARGS.dir)


if not ARGS.extend:
    # Create Directory
    DIR.mkdir()

    # Write arguments
    with DIR.args.open("w") as f:
        f.write(ARGS.to_json())

DIR.run.touch()
