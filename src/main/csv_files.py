from typing import Unpack

from src.random import SeedLike

from .abstract_task import AbstractTask
from .csv_file import CSVFields, CSVFile


class TaskFields(CSVFields):
    Task: AbstractTask
    Time: float
    Seed: SeedLike | None


class TaskCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[TaskFields]): ...
