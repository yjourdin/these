from ...csv_file import CSVFields, CSVFile
from ..elicitation.config import MIPConfig
from .fields import GroupParameters, SRMPParametersDeviation


# Group parameters
class GroupParametersFields(CSVFields):
    Id: int
    Gen: SRMPParametersDeviation
    Accept: SRMPParametersDeviation


class GroupParametersCSVFile(CSVFile[GroupParametersFields]):
    fields = GroupParametersFields


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
    Config: MIPConfig
    Path: bool
    Mie: bool
    P_id: int
    Mie_id: int
    Mc_id: int


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


class MieCSVFile(CSVFile[MieFields]):
    fields = MieFields


# Iteration
class IterationFields(ExperimentFields):
    It: int


# DM Iteration
class DMFields(IterationFields):
    Dm_id: int


# MIP
class CollectiveFields(IterationFields):
    Time: float
    Fitness: float | None


class CollectiveCSVFile(CSVFile[CollectiveFields]):
    fields = CollectiveFields


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
