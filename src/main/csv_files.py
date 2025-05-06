from ..random import Seed
from .abstract_task import AbstractTask
from .csv_file import CSVFile, CSVFields


# Task
class TaskFields(CSVFields):
    Task: AbstractTask
    Time: float
    Seed: Seed | None


class TaskCSVFile(CSVFile[TaskFields]):
    fields = TaskFields
