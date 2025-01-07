from collections.abc import Sequence
from typing import cast

from mcda import PerformanceTable
from mcda.internal.core.relations import Relation
from mcda.relations import PreferenceStructure

from ..model import FrozenModel
from ..preference_structure.generate import preference_relation_generator


def preference_path(
    path: Sequence[FrozenModel],
    alternatives: PerformanceTable,
    start_preferences: PreferenceStructure,
):
    result: list[PreferenceStructure] = []

    for model in path:
        pref_struct = PreferenceStructure()
        pref_struct._relations = cast(
            list[Relation],
            list(
                preference_relation_generator(
                    model.model.rank(alternatives),
                    start_preferences.elements_pairs_relations,
                )
            ),
        )

        result.append(pref_struct)

    return result


def remove_reverted_changes(preference_path: list[PreferenceStructure]):
    preference_path_set = [set(preference._relations) for preference in preference_path]
    i = 2
    while i < len(preference_path):
        changes = preference_path_set[i] - preference_path_set[i - 1]
        for j in reversed(range(i - 2)):
            if changes & preference_path_set[j]:
                del preference_path[j + 1 : i]
                del preference_path_set[j + 1 : i]
                i = j + 1
                break
        i += 1
