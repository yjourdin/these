from ..random import SeedLike
from .abstract_task import AbstractTask
from .csv_file import CSVFile, CSVFields


# Task
class TaskFields(CSVFields):
    Task: AbstractTask
    Time: float
    Seed: SeedLike | None


class TaskCSVFile(CSVFile[TaskFields]):
    fields = TaskFields
