from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import Queue
from pathlib import Path
from threading import Thread
from typing import ClassVar, TypedDict

from ..dataclass import FrozenDataclass
from ..utils import dict_str
from .threads.csv_file import csv_file_thread


class CSVFields(TypedDict): ...


@dataclass(frozen=True)
class CSVFile[Fields: CSVFields](FrozenDataclass):
    path: Path
    fields: ClassVar[type[Fields]]  # type: ignore
    queue: Queue = field(default_factory=Queue)
    thread: Thread = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "thread",
            Thread(target=csv_file_thread, args=(self.path, self.fieldnames, self.queue)),
        )

    @cached_property
    def fieldnames(self):
        return self.fields.__annotations__.keys()

    def writerow(self, fields: Fields):
        self.queue.put(dict_str(fields))
