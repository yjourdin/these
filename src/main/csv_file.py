from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import Queue
from pathlib import Path
from typing import ClassVar, TypedDict

from ..dataclass import FrozenDataclass
from ..utils import dict_str


class CSVFields(TypedDict): ...


@dataclass(frozen=True)
class CSVFile[Fields: CSVFields](FrozenDataclass):
    path: Path
    fields: ClassVar[type[Fields]]  # type: ignore
    queue: "Queue[dict[str, str]]" = field(default_factory=Queue)

    @cached_property
    def fieldnames(self):
        return list(self.fields.__annotations__.keys())

    def writerow(self, fields: Fields):
        self.queue.put(dict_str(fields))

    def close(self):
        self.queue.put({})
