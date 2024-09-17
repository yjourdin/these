from itertools import permutations, product
from typing import NamedTuple, cast

from mcda.relations import I, P, PreferenceStructure
from pulp import value

from ..constants import DEFAULT_MAX_TIME
from ..model import Model
from ..models import GroupModelEnum, ModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..srmp.model import SRMPParamEnum
from .formulation.srmp import MIPSRMP
from .formulation.srmp_group import MIPSRMPGroup
from .formulation.srmp_group_lexicographic import MIPSRMPGroupLexicographicOrder
from .mip import MIP


class MIPResult(NamedTuple):
    best_model: Model | None = None
    best_fitness: float = 0
    time: float = 0


def learn_mip(
    model_type: GroupModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    max_time: int = DEFAULT_MAX_TIME,
    *args,
    **kwargs,
):
    NB_DM = len(comparisons)
    DMS = range(NB_DM)

    alternatives = alternatives.subtable(
        list(set.union(*map(set, (comparisons[dm].elements for dm in DMS))))
    )

    best_model = None
    best_fitness: float = 0
    time = 0
    time_left = max_time

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

    lex_order_shared = (SRMPParamEnum.LEXICOGRAPHIC_ORDER in model_type.value[1]) or (
        NB_DM == 1
    )

    if model_type.value[0] is ModelEnum.SRMP:
        shared_params = cast(set[SRMPParamEnum], model_type.value[1])

        if NB_DM == 1:
            preference_relations = preference_relations_list[0]
            indifference_relations = indifference_relations_list[0]

        for lexicographic_order in (
            permutations(range(k))
            if lex_order_shared
            else product(permutations(range(k)), repeat=NB_DM)
        ):
            if time_left >= 1:
                mip: MIP
                if lex_order_shared:
                    lexicographic_order = cast(tuple[int, ...], lexicographic_order)
                    if NB_DM == 1:
                        mip = MIPSRMP(
                            alternatives,
                            preference_relations,
                            indifference_relations,
                            lexicographic_order,
                            time_limit=time_left,
                            *args,
                            **kwargs,
                        )
                    else:
                        mip = MIPSRMPGroupLexicographicOrder(
                            alternatives,
                            preference_relations_list,
                            indifference_relations_list,
                            lexicographic_order,
                            shared_params,
                            time_limit=time_left,
                            *args,
                            **kwargs,
                        )
                else:
                    lexicographic_order = cast(
                        tuple[tuple[int, ...]], lexicographic_order
                    )
                    mip = MIPSRMPGroup(
                        alternatives,
                        preference_relations_list,
                        indifference_relations_list,
                        lexicographic_order,
                        shared_params,
                        time_limit=time_left,
                        *args,
                        **kwargs,
                    )

                model = mip.learn()

                time += mip.prob.solutionCpuTime
                time_left = max_time - time
                status = mip.prob.status
                objective = mip.prob.objective

                if model is not None:
                    fitness = (
                        (
                            cast(int, value(objective))
                            / sum(len(comparisons[dm]) for dm in DMS)
                            if status > 0
                            else 0
                        )
                        if objective
                        else 1
                    )

                    if fitness > best_fitness:
                        best_model = model
                        best_fitness = fitness

                        if best_fitness == 1:
                            break

        return MIPResult(best_model, best_fitness, time)
    return MIPResult()
