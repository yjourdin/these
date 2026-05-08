from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import JoinableQueue
from pathlib import Path
from typing import Any, TypedDict

from src.dataclass import FrozenDataclass
from src.utils import dict_str


class CSVFields(TypedDict): ...

@dataclass(frozen=True)
class CSVFile(FrozenDataclass):
    path: Path
    queue: "JoinableQueue[dict[str, str]]" = field(default_factory=JoinableQueue)

    @cached_property
    def fieldnames(self):
        return list(CSVFields.__annotations__.keys())

    def writerow(self, **kwargs: Any):
        self.queue.put(dict_str(kwargs))

    def close(self):
        self.queue.put({})
        self.queue.join()
