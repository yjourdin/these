from functools import cached_property
from typing import Unpack

from src.random import SeedLike

from .abstract_task import AbstractTask
from .csv_file import CSVFields, CSVFile


class TaskFields(CSVFields):
    Task: AbstractTask
    Time: float
    Seed: SeedLike | None


class TaskCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(TaskFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[TaskFields]):
        return super().writerow(**kwargs)
