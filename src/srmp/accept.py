from mcda.relations import PreferenceStructure

from ..mip.main import learn_mip
from ..models import GroupModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .model import SRMPModel


def accept(
    model: SRMPModel,
    alternatives: NormalPerformanceTable,
    preferences: PreferenceStructure,
    profiles_amp: float,
    weights_amp: float,
):
    return learn_mip(
        GroupModelEnum.SRMP,
        len(model.profiles.alternatives),
        alternatives,
        [preferences],
        reference_model=model,
        profiles_amp=profiles_amp,
        weights_amp=weights_amp,
    )
