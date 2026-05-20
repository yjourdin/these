from pathlib import Path

from src.case_insensitive_str_enum import CaseInsensitiveStrEnum
from src.dataclass import Dataclass, dataclass
from src.default_max_jobs import DEFAULT_MAX_JOBS

from .directory import RESULTS_DIR


class ExperimentEnum(CaseInsensitiveStrEnum):
    ELICITATION = "E"
    GROUP_DECISION = "G"


@dataclass(init=False, kw_only=True)
class Arguments(Dataclass):
    dir: Path = RESULTS_DIR
    name: str = ""
    jobs: int = DEFAULT_MAX_JOBS
    stop_error: bool = False
    extend: bool = False
