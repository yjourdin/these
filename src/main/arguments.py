from pathlib import Path

from src.dataclass import Dataclass, dataclass
from src.default_max_jobs import DEFAULT_MAX_JOBS
from src.strenum import StrEnumCustom

from .directory import RESULTS_DIR


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
