from typing import Any, NotRequired

from src.methods import MethodEnum
from src.models import GroupModelEnum

from ...csv_file import CSVFields, CSVFile
from .config import Config


# Config
class ConfigFields(CSVFields):
    Id: int
    Method: MethodEnum
    Config: dict[str, Any]


class ConfigCSVFile(CSVFile[ConfigFields]):
    fields = ConfigFields


# Experiment
class ExperimentFields(CSVFields):
    M: int
    N_tr: int
    Atr_id: int
    Mo: GroupModelEnum
    Ko: int
    Group_size: int
    Mo_id: int
    N_bc: int
    Same_alt: bool
    Error: float
    D_id: int
    Me: GroupModelEnum
    Ke: int
    Method: MethodEnum
    Config: Config
    Me_id: int


# Train
class TrainFields(ExperimentFields):
    Time: float
    Fitness: float | None
    It: NotRequired[int]


class TrainCSVFile(CSVFile[TrainFields]):
    fields = TrainFields


# Test
class TestFields(ExperimentFields):
    N_te: int
    Ate_id: int
    Name: str
    Value: float


class TestCSVFile(CSVFile[TestFields]):
    fields = TestFields
