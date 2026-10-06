from collections.abc import Sequence

from mcda.relations import PreferenceStructure

from src.constants import DEFAULT_MAX_TIME
from src.model import FrozenModel, Model
from src.performance_table.type import PerformanceTableType
from src.preference_structure.fitness import (
    fitness_comparisons_ranking,
)
from src.random import RNGParam
from src.rmp.model import FrozenRMPModel, RMPModel
from src.srmp.model import FrozenSRMPModel, SRMPModel

from .a_star import Astar
from .gbfs import GBFS
from .neighborhood import (
    Neighborhood,
    NeighborhoodCombined,
    NeighborhoodImportanceRelation,
    NeighborhoodLexOrder,
    NeighborhoodProfile,
    NeighborhoodWeight,
)
from .preference_path import preference_path, remove_refused, remove_reverted_changes


def compute_model_paths(
    start_models: list[RMPModel | SRMPModel],
    target_preferences: PreferenceStructure,
    alternatives: PerformanceTableType,
    rng: RNGParam = None,
    max_time: int = DEFAULT_MAX_TIME,
    fixed_lex_order: bool = False,
):
    alternatives = alternatives.subtable(target_preferences.elements)

    neighborhoods: list[Neighborhood[FrozenRMPModel | FrozenSRMPModel]] = [
        NeighborhoodProfile(alternatives, target_preferences),
    ]

    if isinstance(start_models[0], RMPModel):
        neighborhoods.append(NeighborhoodImportanceRelation(alternatives, target_preferences))
    else:
        neighborhoods.append(NeighborhoodWeight(alternatives, target_preferences))

    if not fixed_lex_order:
        neighborhoods.append(NeighborhoodLexOrder())

    neighborhood = NeighborhoodCombined(neighborhoods, rng)

    def heuristic(model: FrozenRMPModel | FrozenSRMPModel):
        return int((1 - fitness_comparisons_ranking(
            target_preferences, model.model.rank_series(alternatives)
        )) * len(target_preferences))

    gbfs = GBFS(neighborhood, heuristic, max_time)
    path = gbfs([model.frozen for model in start_models])

    return path, gbfs.time


def compute_preference_path(
    model_path: Sequence[FrozenModel[Model]],
    start_preferences: PreferenceStructure,
    alternatives: PerformanceTableType,
    refused: PreferenceStructure | None = None,
):
    path = preference_path(model_path, alternatives, start_preferences)

    if refused:
        remove_refused(path, refused)
    remove_reverted_changes(path)

    return path
