from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.field import Field, field

from .directory import RESULTS_DIR


@field("dir")
@dataclass
class DirField(Field):
    dir: Path = RESULTS_DIR

    @staticmethod
    def field_decode(o: Any):
        return Path(o)
