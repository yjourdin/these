from typing import TypedDict

from ...csv_file import CSVFile
from ..elicitation.config import MIPConfig
from .fields import GroupParameters, SRMPParametersDeviation


# Group parameters
class GroupParametersFields(TypedDict):
    Id: int
    Gen: SRMPParametersDeviation
    Accept: SRMPParametersDeviation


class GroupParametersCSVFile(CSVFile[GroupParametersFields]):
    fields = GroupParametersFields


# Experiment
class ExperimentFields(TypedDict):
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
    Mc_id: int
    Path: bool
    P_id: int


# Iteration
class IterationFields(ExperimentFields):
    It: int


# DM Iteration
class DMFields(IterationFields):
    Dm_id: int


# MIP
class MIPFields(IterationFields):
    Time: float
    Fitness: float | None


class MIPCSVFile(CSVFile[MIPFields]):
    fields = MIPFields


# Path
class PathFields(DMFields):
    Time: float
    Length: int | None
    Model_Length: int


class PathCSVFile(CSVFile[PathFields]):
    fields = PathFields


# Accept
class AcceptFields(DMFields):
    T: int | None
    Accept: bool


class AcceptCSVFile(CSVFile[AcceptFields]):
    fields = AcceptFields


# Changes
class ChangesFields(DMFields):
    T: int | None
    Changes: int


class ChangesCSVFile(CSVFile[ChangesFields]):
    fields = ChangesFields


# Clean
class CleanFields(DMFields):
    Removed: int
    Total: int


class CleanCSVFile(CSVFile[CleanFields]):
    fields = CleanFields


# Compromise
class CompromiseFields(ExperimentFields):
    Compromise: bool
    Time: float
    It: int
    Changes: int


class CompromiseCSVFile(CSVFile[CompromiseFields]):
    fields = CompromiseFields
