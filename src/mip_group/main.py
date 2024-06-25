from itertools import permutations
from typing import NamedTuple, cast

from mcda.relations import I, P, PreferenceStructure
from pulp import value

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..srmp.model import SRMPModel
from .mip import MIPGroup


class MIPGroupResult(NamedTuple):
    best_models: list[SRMPModel] | None
    best_fitness: float
    time: float | None


def learn_mip(
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    gamma: float = 0.001,
    inconsistencies: bool = True,
    seed: int = 0,
    verbose: bool = False,
):
    alternatives = alternatives.subtable(
        list(set().union(comp.elements for comp in comparisons))
    )

    best_models = None
    best_fitness: float = 0
    time = None

    preference_relations = []
    indifference_relations = []
    for comp in comparisons:
        pref_rels = PreferenceStructure()
        indiff_rels = PreferenceStructure()
        for r in comp:
            match r:
                case P():
                    pref_rels._relations.append(r)
                case I():
                    indiff_rels._relations.append(r)
        preference_relations.append(pref_rels)
        indifference_relations.append(indiff_rels)

    for lexicographic_order in permutations(range(k)):
        mip = MIPGroup(
            alternatives,
            preference_relations,
            indifference_relations,
            lexicographic_order,
            gamma=gamma,
            inconsistencies=inconsistencies,
            seed=seed,
            verbose=verbose,
        )
        models = mip.learn()
        if models is not None:
            objective = mip.prob.objective
            fitness = (
                cast(int, value(objective)) / (sum(len(comp) for comp in comparisons))
                if objective
                else 1
            )

            if fitness > best_fitness:
                best_models = models
                best_fitness = fitness
                time = mip.prob.solutionCpuTime

                if best_fitness == 1:
                    break

    return MIPGroupResult(best_models, best_fitness, time)
