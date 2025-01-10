from collections.abc import Container, Sequence

from mcda import PerformanceTable
from mcda.relations import PreferenceStructure
from numpy.random import Generator

from ..constants import DEFAULT_MAX_TIME
from ..model import FrozenModel
from ..preference_structure.fitness import fitness_comparisons_ranking
from ..random import rng
from ..srmp.model import FrozenSRMPModel, SRMPModel
from .gbfs import GBFS
from .neighborhood import (
    NeighborhoodCombined,
    NeighborhoodLexOrder,
    NeighborhoodProfile,
    NeighborhoodWeight,
)
from .preference_path import preference_path, remove_refused, remove_reverted_changes


def compute_model_path(
    Mc: SRMPModel,
    D: PreferenceStructure,
    A: PerformanceTable,
    rng: Generator = rng(),
    max_time: int = DEFAULT_MAX_TIME,
):
    A = A.subtable(D.elements)

    neighborhood = NeighborhoodCombined(
        [
            NeighborhoodProfile(A),
            NeighborhoodWeight(len(A.criteria)),
            NeighborhoodLexOrder(),
        ],
        rng,
    )

    def heuristic(model: FrozenSRMPModel):
        return 1 - fitness_comparisons_ranking(D, model.model.rank(A))

    gbfs = GBFS(neighborhood, heuristic, max_time)
    path = gbfs(Mc.frozen)

    return path, gbfs.time


def compute_preference_path(
    model_path: Sequence[FrozenModel],
    D: PreferenceStructure,
    A: PerformanceTable,
    refused: Container[PreferenceStructure],
):
    path = preference_path(model_path, A, D)

    remove_refused(path, refused)
    remove_reverted_changes(path)

    return path
