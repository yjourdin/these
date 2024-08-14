from itertools import chain, permutations, product
from typing import NamedTuple, cast

from mcda.relations import I, P, PreferenceStructure
from pulp import value

from ..model import Model
from ..models import ModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..srmp.model import SRMPParamEnum
from .formulation.srmp import MIPSRMP
from .formulation.srmp_group import MIPSRMPGroup
from .formulation.srmp_group_lexicographic import MIPSRMPGroupLexicographicOrder
from .mip import MIP


class MIPResult(NamedTuple):
    best_model: Model | None = None
    best_fitness: float = 0
    time: float | None = None


def learn_mip(
    model_type: ModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    shared_params: set[SRMPParamEnum] = set(),
    *args,
    **kwargs,
):
    NB_DM = len(comparisons)
    DMS = range(NB_DM)

    alternatives = alternatives.subtable(
        list(set(chain(comparisons[dm].elements for dm in DMS)))
    )

    best_model = None
    best_fitness: float = 0
    time = None

    preference_relations_list: list[PreferenceStructure] = []
    indifference_relations_list: list[PreferenceStructure] = []
    for dm in DMS:
        preference_relations_dm = PreferenceStructure()
        indifference_relations_dm = PreferenceStructure()
        for r in comparisons[dm]:
            match r:
                case P():
                    preference_relations_dm._relations.append(r)
                case I():
                    indifference_relations_dm._relations.append(r)
        preference_relations_list.append(preference_relations_dm)
        indifference_relations_list.append(indifference_relations_dm)

    if model_type == ModelEnum.SRMP:
        if NB_DM == 1:
            preference_relations = preference_relations_list[0]
            indifference_relations = indifference_relations_list[0]
            shared_params = set(SRMPParamEnum)

        for lexicographic_order in (
            permutations(range(k))
            if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params
            else product(permutations(range(k)), repeat=NB_DM)
        ):
            mip: MIP
            if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                lexicographic_order = cast(tuple[int, ...], lexicographic_order)
                if NB_DM == 1:
                    mip = MIPSRMP(
                        alternatives,
                        preference_relations,
                        indifference_relations,
                        lexicographic_order,
                        *args,
                        **kwargs,
                    )
                else:
                    mip = MIPSRMPGroupLexicographicOrder(
                        alternatives,
                        preference_relations_list,
                        indifference_relations_list,
                        lexicographic_order,
                        *args,
                        **kwargs,
                    )
            else:
                lexicographic_order = cast(tuple[tuple[int, ...]], lexicographic_order)
                mip = MIPSRMPGroup(
                    alternatives,
                    preference_relations_list,
                    indifference_relations_list,
                    lexicographic_order,
                    *args,
                    **kwargs,
                )

            model = mip.learn()
            if model is not None:
                objective = mip.prob.objective
                fitness = (
                    cast(int, value(objective)) / len(comparisons) if objective else 1
                )

                if fitness > best_fitness:
                    best_model = model
                    best_fitness = fitness
                    time = mip.prob.solutionCpuTime

                    if best_fitness == 1:
                        break

        return MIPResult(best_model, best_fitness, time)
    return MIPResult()
