from collections.abc import Container, Sequence

import numpy as np
import numpy.typing as npt
from mcda.relations import PreferenceStructure

from ..model import FrozenModel, Model
from ..performance_table.type import PerformanceTableType
from ..preference_structure.generate import preference_relation_generator
from ..preference_structure.utils import preference_to_numeric


def preference_path(
    path: Sequence[FrozenModel[Model]],
    alternatives: PerformanceTableType,
    start_preferences: PreferenceStructure,
):
    result: list[PreferenceStructure] = []

    pairs = start_preferences.elements_pairs_relations.keys()

    for model in path:
        result.append(
            PreferenceStructure(
                list(
                    preference_relation_generator(model.model.rank(alternatives), pairs)
                ),
                validate=False,
            )
        )

    return result


def remove_refused(
    path: list[PreferenceStructure], refused: Container[PreferenceStructure]
):
    i = 1
    while i < len(path):
        if path[i] in refused:
            del path[i]
        else:
            i += 1


def remove_reverted_changes(preference_path: list[PreferenceStructure]):
    pairs = set(r.elements for r in preference_path[0]) if preference_path else set()
    changes: list[npt.NDArray[np.int_]] = []
    i = 1
    while i < len(preference_path):
        changes_i = np.array([
            preference_to_numeric(preference_path[i].elements_pairs_relations[p])  # type: ignore
            - preference_to_numeric(
                preference_path[i - 1].elements_pairs_relations[p]  # type: ignore
            )
            for p in pairs
        ])

        if (changes_i == 0).all():
            del preference_path[i]
            continue
        for j in range(i - 1):
            if ((changes[j] * changes_i) < 0).any():
                del preference_path[j + 1 : i]
                del changes[j:]
                i = j + 1
                break
        else:
            i += 1
            changes.append(changes_i)
