from typing import cast

from mcda.internal.core.relations import Relation
from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure

from ..dataclass import Dataclass
from ..model import Model
from ..preference_structure.generate import preference_relation_generator


class PreferencePath(Dataclass):
    parents: dict[Model, list[Model]]
    rankings: dict[Model, Ranking]

    def all_paths(self, model: Model, preferences: PreferenceStructure):
        result = []

        for parent in self.parents[model]:
            pref_struct = PreferenceStructure()
            pref_struct._relations = cast(
                list[Relation],
                list(
                    preference_relation_generator(
                        self.rankings[parent], preferences.elements_pairs_relations
                    )
                ),
            )

            for path in self.all_paths(parent, pref_struct):
                result.append([preferences] + path)

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
