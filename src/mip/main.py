from enum import Enum, member
from functools import partial
from itertools import permutations, product
from typing import Any, NamedTuple, cast

import numpy as np
from mcda.relations import I, P, PreferenceStructure
from pulp import value  # type: ignore

from src.constants import DEFAULT_MAX_TIME
from src.model import Model
from src.models import GroupModelEnum
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.utils import divide_preferences
from src.random import RNGParam, SeedLike, rng_
from src.rmp.permutation import all_max_adjacent_distance
from src.srmp.model import SRMPModel, SRMPParamFlag

from ..utils import add_filename_suffix
from .formulation.srmp import MIPSRMP
from .formulation.srmp_accept import MIPSRMPAccept
from .formulation.srmp_collective import MIPSRMPCollective
from .formulation.srmp_collective_bound import MIPSRMPCollectiveBound
from .formulation.srmp_collective_distance import MIPSRMPCollectiveDistance
from .formulation.srmp_group import MIPSRMPGroup
from .formulation.srmp_group_close import MIPSRMPGroupClose
from .formulation.srmp_group_lexicographic import MIPSRMPGroupLexicographicOrder
from .mip import MIP


class SenseEnum(Enum):
    MAX = member(max)
    MIN = member(min)


class MIPResult[M: Model, O: float](NamedTuple):
    best_model: M | None = None
    best_objective: O | None = None
    time: float = 0
    optimal: bool = False


def create_mip(
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
    comparisons_refused: PreferenceStructure | None = None,
    comparisons_accepted: PreferenceStructure | None = None,
    reference_model: SRMPModel | None = None,
    profiles_amp: float | None = None,
    weights_amp: float | None = None,
    reference_models: list[SRMPModel] | None = None,
    lexicographic_order_distance: int = 0,
    inconsistencies: bool = False,
    nb_cpus: int = 1,
    *args: Any,
    **kwargs: Any,
):
    # if model_type.model is not ModelEnum.SRMP:
    #     return [], SenseEnum.MIN

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

    profiles_amp = profiles_amp or 1
    weights_amp = weights_amp or 1

    preference_relations: list[P] = []
    indifference_relations: list[I] = []
    if NB_DM == 1:
        preference_relations = preference_relations_list[0]
        indifference_relations = indifference_relations_list[0]

    if lex_order:
        lexicographic_orders_array = np.array([lex_order], dtype=np.int_)
    elif lex_order_shared:
        if reference_model is None or lexicographic_order_distance == 0:
            lexicographic_orders_array = np.array(list(permutations(range(k))))
        else:
            lexicographic_orders_array = np.array(
                list(
                    all_max_adjacent_distance(
                        reference_model.lexicographic_order,
                        lexicographic_order_distance,
                    ),
                )
            )

    else:
        lexicographic_orders_array = np.array(
            list(product(permutations(range(k)), repeat=NB_DM))
        )

    rng_(rng_lexicographic_order).shuffle(lexicographic_orders_array)

    lexicographic_orders: list[Any] = lexicographic_orders_array.tolist()

    NB_CPUS_MIP = max(nb_cpus // len(lexicographic_orders), 1)

    sense = SenseEnum.MIN

    if lex_order_shared:
        if NB_DM == 1:
            if reference_model:
                mips = partial(
                    MIPSRMPAccept,
                    alternatives=alternatives,
                    preference_relations=preference_relations,
                    indifference_relations=indifference_relations,
                    model=reference_model,
                    profiles_amp=profiles_amp,
                    weights_amp=weights_amp,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )

            else:
                mips = partial(
                    MIPSRMP,
                    alternatives=alternatives,
                    preference_relations=preference_relations,
                    indifference_relations=indifference_relations,
                    inconsistencies=inconsistencies,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
                sense = SenseEnum.MAX
        elif collective:
            comparisons_refused = comparisons_refused or PreferenceStructure()
            preference_refused_list, indifference_refused_list = divide_preferences(
                comparisons_refused
            )

            # preference_to_accept_list: list[list[P]] = []
            # indifference_to_accept_list: list[list[I]] = []
            # for comp in comparisons_refused:
            #     comp_complementary = complementary_preference(comp)

            #     preference_to_accept, indifference_to_accept = divide_preferences(
            #         comp_complementary
            #     )
            #     preference_to_accept_list.append(preference_to_accept)
            #     indifference_to_accept_list.append(indifference_to_accept)

            comparisons_accepted = comparisons_accepted or PreferenceStructure()
            preference_accepted_list, indifference_accepted_list = divide_preferences(
                comparisons_accepted
            )

            if reference_model:
                mips = partial(
                    MIPSRMPCollectiveDistance,
                    alternatives=alternatives,
                    preference_relations=preference_relations_list,
                    indifference_relations=indifference_relations_list,
                    preferences_changed=preferences_changes,
                    preference_refused=preference_refused_list,
                    indifference_refused=indifference_refused_list,
                    preference_accepted=preference_accepted_list,
                    indifference_accepted=indifference_accepted_list,
                    model=reference_model,
                    profiles_amp=profiles_amp,
                    weights_amp=weights_amp,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
            elif reference_models:
                mips = partial(
                    MIPSRMPCollectiveBound,
                    alternatives=alternatives,
                    preference_relations=preference_relations_list,
                    indifference_relations=indifference_relations_list,
                    preferences_changed=preferences_changes,
                    preference_refused=preference_refused_list,
                    indifference_refused=indifference_refused_list,
                    preference_accepted=preference_accepted_list,
                    indifference_accepted=indifference_accepted_list,
                    models=reference_models,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
            else:
                mips = partial(
                    MIPSRMPCollective,
                    alternatives=alternatives,
                    preference_relations=preference_relations_list,
                    indifference_relations=indifference_relations_list,
                    preferences_changed=preferences_changes,
                    preference_refused=preference_refused_list,
                    indifference_refused=indifference_refused_list,
                    preference_accepted=preference_accepted_list,
                    indifference_accepted=indifference_accepted_list,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
        else:
            if close:
                mips = partial(
                    MIPSRMPGroupClose,
                    alternatives=alternatives,
                    preference_relations=preference_relations_list,
                    indifference_relations=indifference_relations_list,
                    inconsistencies=inconsistencies,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
            else:
                mips = partial(
                    MIPSRMPGroupLexicographicOrder,
                    alternatives=alternatives,
                    preference_relations=preference_relations_list,
                    indifference_relations=indifference_relations_list,
                    shared_params=shared_params,
                    inconsistencies=inconsistencies,
                    time_limit=max_time,
                    seed=seed_mip,
                    nb_cpus=NB_CPUS_MIP,
                )
                sense = SenseEnum.MAX
    else:
        mips = partial(
            MIPSRMPGroup,
            alternatives=alternatives,
            preference_relations=preference_relations_list,
            indifference_relations=indifference_relations_list,
            shared_params=shared_params,
            inconsistencies=inconsistencies,
            time_limit=max_time,
            seed=seed_mip,
            nb_cpus=NB_CPUS_MIP,
        )
        sense = SenseEnum.MAX

    mips = (
        partial(mips, lexicographic_order=lexicographic_order)
        for lexicographic_order in lexicographic_orders
    )

    if log_path := kwargs.pop("log_path", None):
        mips = (
            partial(mip, log_path=add_filename_suffix(log_path, f"_{i}"))
            for i, mip in enumerate(mips)
        )

    return ((mip(**kwargs) for mip in mips), sense)

    # with ThreadPoolExecutor(NB_WORKERS) as thread_pool:
    #     tic = monotonic()
    #     result = MIPResult[Model, float]()
    #     try:
    #         best_model, best_objective, _ = sense(
    #             thread_pool.map(mip_result, mips, timeout=max_time),
    #             key=attrgetter("best_objective"),
    #         )

    #         result = result._replace(
    #             best_model=best_model, best_objective=best_objective
    #         )
    #     except (TimeoutError, TypeError):
    #         ...
    #     toc = monotonic()
    #     result = result._replace(time=toc - tic)
    #     return result


def mip_result[M: Model](mip: MIP[M, Any, Any]):
    best_sol = mip.learn()
    best_objective = (
        cast(float, value(objective))
        if (objective := mip.prob.objective) is not None
        else None
    )
    return MIPResult[M, float](
        best_sol if best_objective is not None else None,
        best_objective if best_objective is not None else None,
        mip.prob.solutionCpuTime,
        mip.prob.sol_status == 1,
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
