from functools import cached_property
from typing import Any, NotRequired, Unpack

from src.methods import MethodEnum
from src.models import GroupModelEnum

from ...csv_file import CSVFields, CSVFile
from .config import Config


# Config
class ConfigFields(CSVFields):
    Id: int
    Method: MethodEnum
    Config: dict[str, Any]


class ConfigCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(ConfigFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[ConfigFields]):
        return super().writerow(**kwargs)


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


class TrainCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(TrainFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[TrainFields]):
        return super().writerow(**kwargs)


# Test
class TestFields(ExperimentFields):
    N_te: int
    Ate_id: int
    Name: str
    Value: float


class TestCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(TestFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[TestFields]):
        return super().writerow(**kwargs)
