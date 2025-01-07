from collections import defaultdict
from itertools import permutations, product
from typing import NamedTuple, cast

import numpy as np
from mcda.relations import PreferenceStructure
from numpy.random import Generator
from pulp import value

from src.preference_structure.utils import divide_preferences
from src.rmp.permutation import all_max_adjacent_distance

from ..constants import DEFAULT_MAX_TIME
from ..model import Model
from ..models import GroupModelEnum, ModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..random import Seed
from ..srmp.model import SRMPModel, SRMPParamEnum
from .formulation.srmp import MIPSRMP
from .formulation.srmp_accept import MIPSRMPAccept
from .formulation.srmp_collective import MIPSRMPCollective
from .formulation.srmp_group import MIPSRMPGroup
from .formulation.srmp_group_lexicographic import MIPSRMPGroupLexicographicOrder
from .mip import MIP


class MIPResult(NamedTuple):
    best_model: Model | None = None
    best_fitness: float | None = None
    time: float = 0


def learn_mip(
    model_type: GroupModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    rng_lexicographic_order: Generator,
    seed_mip: Seed,
    max_time: int = DEFAULT_MAX_TIME,
    collective: bool = False,
    preferences_changed: list[int] | None = None,
    comparisons_refused: list[list[PreferenceStructure]] | None = None,
    reference_model: SRMPModel | None = None,
    profiles_amp: float = 1,
    weights_amp: float = 1,
    lexicographic_order_distance: int = 0,
    *args,
    **kwargs,
):
    if model_type.value[0] is not ModelEnum.SRMP:
        return MIPResult()

    NB_DM = len(comparisons)
    DMS = range(NB_DM)

    alternatives = alternatives.subtable(
        list(set.union(*map(set, (comparisons[dm].elements for dm in DMS))))
    )

    best_model = None
    best_objective: float | None = (
        None  # = sum(len(comp) for comp in comparisons) if collective else 0
    )
    time = 0
    time_left = max_time

    preference_relations_list: list[PreferenceStructure] = []
    indifference_relations_list: list[PreferenceStructure] = []
    for dm in DMS:
        preference_relations_dm, indifference_relations_dm = divide_preferences(
            comparisons[dm]
        )
        preference_relations_list.append(preference_relations_dm)
        indifference_relations_list.append(indifference_relations_dm)

    lex_order_shared = (
        (SRMPParamEnum.LEXICOGRAPHIC_ORDER in model_type.value[1])
        or (NB_DM == 1)
        or collective
    )

    preferences_changed = preferences_changed or ([0] * NB_DM)

    shared_params = cast(set[SRMPParamEnum], model_type.value[1])

    if NB_DM == 1:
        preference_relations = preference_relations_list[0]
        indifference_relations = indifference_relations_list[0]

    if lex_order_shared:
        if reference_model is None or lexicographic_order_distance == 0:
            lexicographic_orders = np.array(list(permutations(range(k))))
        else:
            lexicographic_orders = np.array(
                list(
                    product(
                        all_max_adjacent_distance(
                            reference_model.lexicographic_order,
                            lexicographic_order_distance,
                        ),
                        repeat=NB_DM,
                    )
                )
            )
    else:
        lexicographic_orders = np.array(
            list(product(permutations(range(k)), repeat=NB_DM))
        )

    rng_lexicographic_order.shuffle(lexicographic_orders)

    for lexicographic_order in lexicographic_orders:
        if time_left >= 1:
            mip: MIP
            if lex_order_shared:
                lexicographic_order = cast(list[int], lexicographic_order.tolist())
                if NB_DM == 1:
                    if reference_model:
                        mip = MIPSRMPAccept(
                            alternatives,
                            preference_relations,
                            indifference_relations,
                            lexicographic_order,
                            reference_model,
                            profiles_amp,
                            weights_amp,
                            time_limit=time_left,
                            seed=seed_mip,
                            *args,
                            **kwargs,
                        )
                    else:
                        mip = MIPSRMP(
                            alternatives,
                            preference_relations,
                            indifference_relations,
                            lexicographic_order,
                            best_fitness=best_objective,
                            time_limit=time_left,
                            seed=seed_mip,
                            *args,
                            **kwargs,
                        )
                elif collective:
                    assert comparisons_refused is not None
                    comparisons_refused_list = []
                    count_refused_dict = defaultdict(int)
                    for comp_dm in comparisons_refused:
                        count_refused_dict_dm = defaultdict(int)
                        for comp in comp_dm:
                            if comp not in comparisons_refused_list:
                                comparisons_refused_list.append(comp)
                            count_refused_dict_dm[
                                comparisons_refused_list.index(comp)
                            ] += 1
                        for comp, count in count_refused_dict_dm.items():
                            count_refused_dict[comp] = max(
                                count_refused_dict[comp], count
                            )
                    preference_refused_list = []
                    indifference_refused_list = []
                    for comp in comparisons_refused_list:
                        preference_refused, indifference_refused = divide_preferences(
                            comp
                        )
                        preference_refused_list.append(preference_refused)
                        indifference_refused_list.append(indifference_refused)
                    count_refused_list = list(count_refused_dict.values())

                    mip = MIPSRMPCollective(
                        alternatives,
                        preference_relations_list,
                        indifference_relations_list,
                        lexicographic_order,
                        preferences_changed,
                        preference_refused_list,
                        indifference_refused_list,
                        count_refused_list,
                        best_objective=best_objective,
                        time_limit=time_left,
                        seed=seed_mip,
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
                        best_fitness=best_objective,
                        time_limit=time_left,
                        *args,
                        **kwargs,
                    )
            else:
                lexicographic_order = cast(
                    list[list[int]], lexicographic_order.tolist()
                )
                mip = MIPSRMPGroup(
                    alternatives,
                    preference_relations_list,
                    indifference_relations_list,
                    lexicographic_order,
                    shared_params,
                    best_fitness=best_objective,
                    time_limit=time_left,
                    seed=seed_mip,
                    *args,
                    **kwargs,
                )

            model = mip.learn()

            time += mip.prob.solutionCpuTime
            time_left = max_time - time
            status = mip.prob.status
            objective = mip.prob.objective

            if model is not None:
                if collective:
                    best_model = model
                    best_objective = cast(int, value(objective))

                    if best_objective - sum(preferences_changed) == 0:
                        break
                else:
                    best_model = model
                    best_objective = (
                        (
                            cast(int, value(objective))
                            / sum(len(comparisons[dm]) for dm in DMS)
                            if status > 0
                            else 0
                        )
                        if objective
                        else 1
                    )

                    if best_objective == 1:
                        break

    return MIPResult(best_model, best_objective, time)
