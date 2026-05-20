from .args import ARGS
from .arguments import ExperimentEnum
from .directory import Directory
from .experiments.elicitation.main import main as main_elicitation
from .experiments.group_decision.main import main as main_group_decision

# Set main function
directory_class = Directory
match ARGS.experiment:
    case ExperimentEnum.ELICITATION:
        MAIN = main_elicitation
    case ExperimentEnum.GROUP_DECISION:
        MAIN = main_group_decision  # pyright: ignore[reportConstantRedefinition]
