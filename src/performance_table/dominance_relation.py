from itertools import combinations

from mcda import PerformanceTable
from mcda.relations import P, PreferenceStructure

from src.relation import Relation


def dominance_relation(performance_table: PerformanceTable):
    result = Relation(
        len(performance_table.alternatives), performance_table.alternatives
    )
    alternatives_values = performance_table.alternatives_values
    values = {a: alternatives_values[a] for a in performance_table.alternatives}
    for a, b in combinations(performance_table.alternatives, 2):
        if values[a].dominate(values[b]):
            result.data[a, b] = True
        if values[b].dominate(values[a]):
            result.data[b, a] = True
    return result


def dominance_structure(performance_table: PerformanceTable):
    result = PreferenceStructure()
    alternatives_values = performance_table.alternatives_values
    values = {a: alternatives_values[a] for a in performance_table.alternatives}
    for a, b in combinations(performance_table.alternatives, 2):
        if values[a].dominate(values[b]):
            result._relations.append(P(a, b))
        if values[b].dominate(values[a]):
            result._relations.append(P(b, a))
    return result
