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
    def writerow(self, **kwargs: Unpack[GroupParametersFields]): ...


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
    def writerow(self, **kwargs: Unpack[MieFields]): ...


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


class CollectiveCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[CollectiveFields]): ...


# Path
class PathFields(DMFields):
    Time: float
    Length: int | None
    Model_Length: int | None


class PathCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[PathFields]): ...


# Accept
class AcceptFields(DMFields):
    T: int | None
    Accept: bool


class AcceptCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[AcceptFields]): ...


# Changes
class ChangesFields(DMFields):
    T: int | None
    Changes: int


class ChangesCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[ChangesFields]): ...


# Clean
class CleanFields(DMFields):
    Removed: int
    Total: int


class CleanCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[CleanFields]): ...


# Compromise
class CompromiseFields(ExperimentFields):
    Compromise: bool
    Time: float
    It: int
    Changes: int


class CompromiseCSVFile(CSVFile):
    def writerow(self, **kwargs: Unpack[CompromiseFields]): ...
