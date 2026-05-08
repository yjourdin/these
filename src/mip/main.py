from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
from operator import attrgetter
from time import monotonic
from typing import Any, NamedTuple, cast

import numpy as np
from mcda.relations import I, P, PreferenceStructure
from pulp import value  # type: ignore

from src.constants import DEFAULT_MAX_TIME
from src.model import Model
from src.models import GroupModelEnum, ModelEnum
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.utils import complementary_preference, divide_preferences
from src.random import RNGParam, SeedLike, rng_
from src.rmp.permutation import all_max_adjacent_distance
from src.srmp.model import SRMPModel, SRMPParamFlag

from .formulation.srmp import MIPSRMP
from .formulation.srmp_accept import MIPSRMPAccept
from .formulation.srmp_collective import MIPSRMPCollective
from .formulation.srmp_collective_bound import MIPSRMPCollectiveBound
from .formulation.srmp_collective_distance import MIPSRMPCollectiveDistance
from .formulation.srmp_group import MIPSRMPGroup
from .formulation.srmp_group_close import MIPSRMPGroupClose
from .formulation.srmp_group_lexicographic import MIPSRMPGroupLexicographicOrder
from .mip import MIP


class MIPResult[M: Model, O: float](NamedTuple):
    best_model: M | None = None
    best_objective: O | None = None
    time: float = 0


def learn_mip(
    model_type: GroupModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    rng_lexicographic_order: RNGParam = None,
    seed_mip: SeedLike | None = None,
    max_time: int = DEFAULT_MAX_TIME,
    lex_order: list[int] | None = None,
    collective: bool = False,
    close: bool = False,
    preferences_changes: list[int] | None = None,
    comparisons_refused: list[PreferenceStructure] | None = None,
    comparisons_accepted: PreferenceStructure | None = None,
    reference_model: SRMPModel | None = None,
    profiles_amp: float = 1,
    weights_amp: float = 1,
    reference_models: list[SRMPModel] | None = None,
    lexicographic_order_distance: int = 0,
    inconsistencies: bool = False,
    nb_cpus: int = 1,
    *args: Any,
    **kwargs: Any,
):
    if model_type.model is not ModelEnum.SRMP:
        return MIPResult[Model, float]()

    NB_DM = len(comparisons)
    DMS = range(NB_DM)

    alternatives = alternatives.subtable(
        list(set.union(*(set(comparisons[dm].elements) for dm in DMS)))  # type: ignore
    )

    preference_relations_list: list[list[P]] = []
    indifference_relations_list: list[list[I]] = []
    for dm in DMS:
        preference_relations_dm, indifference_relations_dm = divide_preferences(
            comparisons[dm].relations
        )
        preference_relations_list.append(preference_relations_dm)
        indifference_relations_list.append(indifference_relations_dm)

    lex_order_shared = (
        (SRMPParamFlag.LEXICOGRAPHIC_ORDER in model_type.shared_params)
        or (NB_DM == 1)
        or collective
        or lex_order
        or close
    )

    preferences_changes = preferences_changes or ([0] * NB_DM)

    shared_params = SRMPParamFlag(model_type.shared_params)

    preference_relations: list[P] = []
    indifference_relations: list[I] = []
    if NB_DM == 1:
        preference_relations = preference_relations_list[0]
        indifference_relations = indifference_relations_list[0]

    if lex_order:
        lexicographic_orders = np.array([lex_order], dtype=np.int_)
    elif lex_order_shared:
        if reference_model is None or lexicographic_order_distance == 0:
            lexicographic_orders = np.array(list(permutations(range(k))))
        else:
            lexicographic_orders = np.array(
                list(
                    all_max_adjacent_distance(
                        reference_model.lexicographic_order,
                        lexicographic_order_distance,
                    ),
                )
            )

    else:
        lexicographic_orders = np.array(
            list(product(permutations(range(k)), repeat=NB_DM))
        )

    rng_(rng_lexicographic_order).shuffle(lexicographic_orders)

    lexicographic_orders = lexicographic_orders.tolist()

    NB_CPUS = max(nb_cpus // len(lexicographic_orders), 1)

    sense = min

    if lex_order_shared:
        if NB_DM == 1:
            if reference_model:
                mips = [
                    MIPSRMPAccept(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations,
                        indifference_relations=indifference_relations,
                        lexicographic_order=lexicographic_order,
                        model=reference_model,
                        profiles_amp=profiles_amp,
                        weights_amp=weights_amp,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]

            else:
                mips = [
                    MIPSRMP(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations,
                        indifference_relations=indifference_relations,
                        lexicographic_order=lexicographic_order,
                        inconsistencies=inconsistencies,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
                sense = max
        elif collective:
            comparisons_refused = comparisons_refused or []
            preference_to_accept_list: list[list[P]] = []
            indifference_to_accept_list: list[list[I]] = []
            for comp in comparisons_refused:
                comp_complementary = complementary_preference(comp)

                preference_to_accept, indifference_to_accept = divide_preferences(
                    comp_complementary
                )
                preference_to_accept_list.append(preference_to_accept)
                indifference_to_accept_list.append(indifference_to_accept)

            comparisons_accepted = comparisons_accepted or PreferenceStructure()
            preference_accepted_list, indifference_accepted_list = divide_preferences(
                comparisons_accepted
            )

            if reference_model:
                mips = [
                    MIPSRMPCollectiveDistance(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations_list,
                        indifference_relations=indifference_relations_list,
                        lexicographic_order=lexicographic_order,
                        preferences_changed=preferences_changes,
                        preference_to_accept=preference_to_accept_list,
                        indifference_to_accept=indifference_to_accept_list,
                        preference_accepted=preference_accepted_list,
                        indifference_accepted=indifference_accepted_list,
                        model=reference_model,
                        profiles_amp=profiles_amp,
                        weights_amp=weights_amp,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
            elif reference_models:
                mips = [
                    MIPSRMPCollectiveBound(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations_list,
                        indifference_relations=indifference_relations_list,
                        lexicographic_order=lexicographic_order,
                        preferences_changed=preferences_changes,
                        preference_to_accept=preference_to_accept_list,
                        indifference_to_accept=indifference_to_accept_list,
                        preference_accepted=preference_accepted_list,
                        indifference_accepted=indifference_accepted_list,
                        models=reference_models,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
            else:
                mips = [
                    MIPSRMPCollective(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations_list,
                        indifference_relations=indifference_relations_list,
                        lexicographic_order=lexicographic_order,
                        preferences_changed=preferences_changes,
                        preference_to_accept=preference_to_accept_list,
                        indifference_to_accept=indifference_to_accept_list,
                        preference_accepted=preference_accepted_list,
                        indifference_accepted=indifference_accepted_list,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
        else:
            if close:
                mips = [
                    MIPSRMPGroupClose(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations_list,
                        indifference_relations=indifference_relations_list,
                        lexicographic_order=lexicographic_order,
                        inconsistencies=inconsistencies,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
            else:
                mips = [
                    MIPSRMPGroupLexicographicOrder(
                        *args,
                        alternatives=alternatives,
                        preference_relations=preference_relations_list,
                        indifference_relations=indifference_relations_list,
                        lexicographic_order=lexicographic_order,
                        shared_params=shared_params,
                        inconsistencies=inconsistencies,
                        time_limit=max_time,
                        seed=seed_mip,
                        nb_cpus=NB_CPUS,
                        **kwargs,
                    )
                    for lexicographic_order in lexicographic_orders
                ]
                sense = max
    else:
        mips = [
            MIPSRMPGroup(
                *args,
                alternatives=alternatives,
                preference_relations=preference_relations_list,
                indifference_relations=indifference_relations_list,
                lexicographic_order=lexicographic_order,
                shared_params=shared_params,
                inconsistencies=inconsistencies,
                time_limit=max_time,
                seed=seed_mip,
                nb_cpus=NB_CPUS,
                **kwargs,
            )
            for lexicographic_order in lexicographic_orders
        ]
        sense = max

    with ProcessPoolExecutor(nb_cpus) as process_pool:
        try:
            tic = monotonic()
            best_model, best_objective, _ = sense(
                process_pool.map(mip_result, mips, timeout=max_time),
                key=attrgetter("best_objective"),
            )
            toc = monotonic()
            return MIPResult(best_model, best_objective, toc - tic)
        except TimeoutError:
            return MIPResult[Model, float]()


def mip_result[M: Model](mip: MIP[M, Any, Any]):
    best_model = mip.learn()
    return MIPResult[M, float](
        best_model, cast(float, value(mip.prob.objective)), mip.prob.solutionCpuTime
    )

    #         model = mip.learn()

    #         time += mip.prob.solutionCpuTime
    #         time_left = max_time - time
    #         status = mip.prob.status
    #         objective = mip.prob.objective

    #         if model is not None:
    #             if collective:
    #                 best_model = model
    #                 best_objective = cast(int, value(objective))

    #                 if best_objective == sum(preferences_changes):
    #                     break
    #             else:
    #                 best_model = model
    #                 best_objective = (
    #                     (
    #                         cast(int, value(objective))
    #                         / sum(len(comparisons[dm]) for dm in DMS)
    #                         if status > 0
    #                         else 0
    #                     )
    #                     if objective
    #                     else 1
    #                 )

    #                 if best_objective == 1:
    #                     break

    # return MIPResult(best_model, best_objective, time)
