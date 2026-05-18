from functools import cached_property
from typing import Unpack

from src.methods import MethodEnum

from ...csv_file import CSVFields, CSVFile
from ..elicitation.config import Config, MIPConfig
from .fields import GroupParameters, SRMPParametersDeviation


# Group parameters
class GroupParametersFields(CSVFields):
    Id: int
    Gen: SRMPParametersDeviation
    Accept: SRMPParametersDeviation


class GroupParametersCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(GroupParametersFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[GroupParametersFields]):
        return super().writerow(**kwargs)


# Experiment
class ExperimentFields(CSVFields):
    M: int
    N_tr: int
    Atr_id: int
    Ko: int
    Mo_id: int
    Group_size: int
    Group: GroupParameters
    Mi_id: int
    N_bc: int
    Same_alt: bool
    D_id: int
    Method: MethodEnum
    Config: Config
    Mie: bool
    Mie_config: MIPConfig | None
    Mie_id: int
    Mc_id: int
    Nb_Mcp: int
    Path: bool
    P_id: int


# MIE
class MieFields(CSVFields):
    M: int
    N_tr: int
    Atr_id: int
    Ko: int
    Mo_id: int
    Group_size: int
    Group: GroupParameters
    Mi_id: int
    N_bc: int
    Same_alt: bool
    D_id: int
    Config: MIPConfig
    Mie_id: int
    Time: float
    Fitness: float | None


class MieCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(MieFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[MieFields]):
        return super().writerow(**kwargs)


# Iteration
class IterationFields(ExperimentFields):
    It: int


# DM Iteration
class DMFields(IterationFields):
    Dm_id: int


# MIP
class CollectiveFields(IterationFields):
    Time: float
    Objective: float | None
    Optimal: bool


class CollectiveCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(CollectiveFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[CollectiveFields]):
        return super().writerow(**kwargs)


# Path
class PathFields(DMFields):
    Time: float
    Length: int | None
    Model_Length: int | None


class PathCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(PathFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[PathFields]):
        return super().writerow(**kwargs)


# Accept
class AcceptFields(DMFields):
    T: int


class AcceptCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(AcceptFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[AcceptFields]):
        return super().writerow(**kwargs)


# Changes
class ChangesFields(DMFields):
    T: int | None
    Changes: int


class ChangesCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(ChangesFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[ChangesFields]):
        return super().writerow(**kwargs)


# Clean
class CleanFields(DMFields):
    Removed: int
    Total: int


class CleanCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(CleanFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[CleanFields]):
        return super().writerow(**kwargs)


# Compromise
class CompromiseFields(ExperimentFields):
    Compromise: bool
    Time: float
    It: int
    Changes: int


class CompromiseCSVFile(CSVFile):
    @cached_property
    def fieldnames(self):
        return list(CompromiseFields.__annotations__.keys())

    def writerow(self, **kwargs: Unpack[CompromiseFields]):
        return super().writerow(**kwargs)
