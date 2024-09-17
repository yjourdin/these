from itertools import combinations
from typing import Any

from mcda import PerformanceTable
from mcda.relations import PreferenceStructure, P

DominanceRelation = set[tuple[Any, Any]]


def dominance_relation(performance_table: PerformanceTable):
    result = PreferenceStructure()
    alternatives_values = performance_table.alternatives_values
    values = {a: alternatives_values[a] for a in performance_table.alternatives}
    for a, b in combinations(performance_table.alternatives, 2):
        if values[a].dominate(values[b]):
            result._relations.append(P(a, b))
        if values[b].dominate(values[a]):
            result._relations.append(P(b, a))
    return result
