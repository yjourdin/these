from itertools import combinations

from mcda.internal.core.relations import Relation
from mcda.relations import P, PreferenceStructure

from .type import PerformanceTableType


def dominance_structure(performance_table: PerformanceTableType):
    result: list[Relation] = []
    alternatives_values = performance_table.alternatives_values
    values = {a: alternatives_values[a] for a in performance_table.alternatives}
    for a, b in combinations(performance_table.alternatives, 2):  # type: ignore
        if values[a].dominate(values[b]):
            result.append(P(a, b))
        if values[b].dominate(values[a]):
            result.append(P(b, a))
    return PreferenceStructure(result, validate=False)


def is_subset(PS1: PreferenceStructure, PS2: PreferenceStructure):
    for r in PS1:
        if r not in PS2:
            return False
    return True
