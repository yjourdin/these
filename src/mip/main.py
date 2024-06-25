from itertools import permutations
from typing import NamedTuple, cast

from mcda.relations import I, P, PreferenceStructure
from pulp import value

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..srmp.model import SRMPModel
from .mip import MIP


class MIPResult(NamedTuple):
    best_model: SRMPModel | None
    best_fitness: float
    time: float | None


def learn_mip(
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: PreferenceStructure,
    gamma: float = 0.001,
    inconsistencies: bool = True,
    seed: int = 0,
    verbose: bool = False,
):
    alternatives = alternatives.subtable(comparisons.elements)

    best_model = None
    best_fitness: float = 0
    time = None

    preference_relations = PreferenceStructure()
    indifference_relations = PreferenceStructure()
    for r in comparisons:
        match r:
            case P():
                preference_relations._relations.append(r)
            case I():
                indifference_relations._relations.append(r)

    for lexicographic_order in permutations(range(k)):
        mip = MIP(
            alternatives,
            preference_relations,
            indifference_relations,
            lexicographic_order,
            gamma=gamma,
            inconsistencies=inconsistencies,
            seed=seed,
            verbose=verbose,
        )
        model = mip.learn()
        if model is not None:
            objective = mip.prob.objective
            fitness = cast(int, value(objective)) / len(comparisons) if objective else 1

            if fitness > best_fitness:
                best_model = model
                best_fitness = fitness
                time = mip.prob.solutionCpuTime

                if best_fitness == 1:
                    break

    return MIPResult(best_model, best_fitness, time)
