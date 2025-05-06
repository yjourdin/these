from dataclasses import dataclass
from pathlib import Path

from ..dataclass import Dataclass
from ..default_max_jobs import DEFAULT_MAX_JOBS
from .directory import RESULTS_DIR
from ..enum import StrEnumCustom


class ExperimentEnum(StrEnumCustom):
    ELICITATION = "E"
    GROUP_DECISION = "G"


@dataclass
class Arguments(Dataclass):
    experiment: ExperimentEnum
    dir: Path = RESULTS_DIR
    name: str = ""
    jobs: int = DEFAULT_MAX_JOBS
    stop_error: bool = False
    extend: bool = False
