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
    Neighborhood,
    NeighborhoodCombined,
    NeighborhoodLexOrder,
    NeighborhoodProfile,
    NeighborhoodWeight,
)
from .preference_path import preference_path, remove_refused, remove_reverted_changes


def compute_model_path(
    start_model: SRMPModel,
    target_preferences: PreferenceStructure,
    alternatives: PerformanceTable,
    rng: Generator = rng(),
    max_time: int = DEFAULT_MAX_TIME,
    fixed_lex_order: bool = False,
):
    alternatives = alternatives.subtable(target_preferences.elements)

    neighborhoods: list[Neighborhood[FrozenSRMPModel]] = [
        NeighborhoodProfile(alternatives),
        NeighborhoodWeight(len(alternatives.criteria)),
    ]

    if not fixed_lex_order:
        neighborhoods.append(NeighborhoodLexOrder())

    neighborhood = NeighborhoodCombined(neighborhoods, rng)

    def heuristic(model: FrozenSRMPModel):
        return 1 - fitness_comparisons_ranking(
            target_preferences, model.model.rank(alternatives)
        )

    gbfs = GBFS(neighborhood, heuristic, max_time)
    path = gbfs(start_model.frozen)

    return path, gbfs.time


def compute_preference_path(
    model_path: Sequence[FrozenModel],
    start_preferences: PreferenceStructure,
    alternatives: PerformanceTable,
    refused: Container[PreferenceStructure],
):
    path = preference_path(model_path, alternatives, start_preferences)

    remove_refused(path, refused)
    remove_reverted_changes(path)

    return path
