from collections.abc import Container, Sequence

from mcda.relations import PreferenceStructure

from ..constants import DEFAULT_MAX_TIME
from ..model import FrozenModel, Model
from ..performance_table.type import PerformanceTableType
from ..preference_structure.fitness import fitness_comparisons_ranking
from ..random import RNGParam
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
    alternatives: PerformanceTableType,
    rng: RNGParam = None,
    max_time: int = DEFAULT_MAX_TIME,
    fixed_lex_order: bool = False,
):
    alternatives = alternatives.subtable(target_preferences.elements)

    neighborhoods: list[Neighborhood[FrozenSRMPModel]] = [
        NeighborhoodProfile(alternatives),
        NeighborhoodWeight(),
    ]

    if not fixed_lex_order:
        neighborhoods.append(NeighborhoodLexOrder())

    neighborhood = NeighborhoodCombined(neighborhoods, rng)

    def heuristic(model: FrozenSRMPModel):
        return 1 - fitness_comparisons_ranking(
            target_preferences, model.model.rank_series(alternatives)
        )

    gbfs = GBFS(neighborhood, heuristic, max_time)
    path = gbfs(start_model.frozen)

    return path, gbfs.time


def compute_preference_path(
    model_path: Sequence[FrozenModel[Model]],
    start_preferences: PreferenceStructure,
    alternatives: PerformanceTableType,
    refused: Container[PreferenceStructure],
):
    path = preference_path(model_path, alternatives, start_preferences)

    remove_refused(path, refused)
    remove_reverted_changes(path)

    return path
