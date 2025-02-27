from typing import NotRequired, TypedDict

from ....methods import MethodEnum
from ....models import GroupModelEnum
from ...csv_file import CSVFile
from .config import Config


# Config
class ConfigFields(TypedDict):
    Id: int
    Method: MethodEnum
    Config: Config


class ConfigCSVFile(CSVFile[ConfigFields]):
    fields = ConfigFields


# Experiment
class ExperimentFields(TypedDict):
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
