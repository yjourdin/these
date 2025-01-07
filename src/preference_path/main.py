from mcda import PerformanceTable
from mcda.relations import PreferenceStructure

from ..preference_structure.fitness import fitness_comparisons_ranking
from ..srmp.model import FrozenSRMPModel, SRMPModel
from .a_star import A_star
from .neighborhood import (
    NeighborhoodCombined,
    NeighborhoodLexOrder,
    NeighborhoodProfile,
    NeighborhoodWeight,
)
from .preference_path import preference_path, remove_reverted_changes


def compute_preference_path(Mc: SRMPModel, D: PreferenceStructure, A: PerformanceTable):
    A = A.subtable(D.elements)

    neighborhood = NeighborhoodCombined(
        [
            NeighborhoodProfile(A),
            NeighborhoodWeight(len(A.criteria)),
            NeighborhoodLexOrder(),
        ]
    )

    def heuristic(model: FrozenSRMPModel):
        return 1 - fitness_comparisons_ranking(D, model.model.rank(A))

    model_paths = A_star(neighborhood)(Mc.frozen, 0, heuristic)
    preference_paths = [preference_path(path, A, D) for path in model_paths]

    max_len = 0
    remove_reverted_changes(preference_paths[0])
    max_path = preference_paths[0]
    for path in preference_paths[1:]:
        if len(path) > max_len:
            remove_reverted_changes(path)
            if len(path) > max_len:
                max_path = path

    return max_path
