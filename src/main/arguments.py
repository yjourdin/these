from dataclasses import dataclass

from src.enum_base import StrEnum

from ..dataclass import Dataclass
from ..default_max_jobs import DEFAULT_MAX_JOBS
from .directory import RESULTS_DIR


class ExperimentEnum(StrEnum):
    ELICITATION = "E"
    GROUP_DECISION = "G"


@dataclass
class Arguments(Dataclass):
    experiment: ExperimentEnum
    dir: str = RESULTS_DIR
    name: str = ""
    jobs: int = DEFAULT_MAX_JOBS
    stop_error: bool = False
    extend: bool = False
