from typing import TypedDict

from ..random import Seed
from .abstract_task import AbstractTask
from .csv_file import CSVFile


# Task
class TaskFields(TypedDict):
    Task: AbstractTask
    Time: float
    Seed: Seed | None


class TaskCSVFile(CSVFile[TaskFields]):
    fields = TaskFields
