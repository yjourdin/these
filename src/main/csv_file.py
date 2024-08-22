from collections import UserDict
from dataclasses import dataclass, field
from multiprocessing import Queue
from pathlib import Path
from threading import Thread

from ..dataclass import FrozenDataclass
from .fieldnames import Fieldnames
from .thread.csv_file import csv_file_thread


@dataclass(frozen=True)
class CSVFile(FrozenDataclass):
    path: Path
    fieldnames: type[Fieldnames]
    queue: Queue = field(default_factory=Queue)
    thread: Thread = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "thread",
            Thread(
                target=csv_file_thread, args=(self.path, self.fieldnames, self.queue)
            ),
        )


class CSVFiles(UserDict[str, CSVFile]):
    @property
    def paths(self):
        return {k: v.path for k, v in self.data.items()}

    @property
    def queues(self):
        return {k: v.queue for k, v in self.data.items()}

    @property
    def threads(self):
        return {k: v.thread for k, v in self.data.items()}
