from enum import Enum, member
from typing import NamedTuple, TextIO

from mcda.relations import PreferenceStructure

from src.constants import DEFAULT_MAX_TIME
from src.models import ModelEnum
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.random import RNGParam, rng_
from src.rmp.model import RMPModel
from src.srmp.model import SRMPModel
from src.utils import midpoints

from ..model import Model
from .cooling_schedule import GeometricSchedule
from .initial_temperature import initial_temperature
from .neighbor import (
    Neighbor,
    NeighborImportanceRelation,
    NeighborLexOrder,
    NeighborProfileDiscretized,
    NeighborWeight,
    NeighborWeightAmp,
    RandomNeighbor,
)
from .objective import CollectiveObjective, FitnessObjective
from .sa import SimulatedAnnealing


class SenseEnum(Enum):
    MAX = member(max)
    MIN = member(min)


class SAResult[M, O](NamedTuple):
    best_model: M
    best_objective: O
    time: float
    it: int


def create_sa(
    model: ModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: list[PreferenceStructure],
    alpha: float,
    amp: float,
    lex_order: list[int] | None = None,
    t0: float | None = None,
    accept: float | None = None,
    L: int = 1,
    Tf: float | None = None,
    max_time: int = DEFAULT_MAX_TIME,
    max_it: int | None = None,
    max_it_non_improving: int | None = None,
    log_file: TextIO | None = None,
    preferences_changes: list[int] | None = None,
    comparisons_refused: list[PreferenceStructure] | None = None,
    rng_init: RNGParam = None,
    rng_sa: RNGParam = None,
    nb_cpus: int = 1,
):
    # DMs
    NB_DM = len(comparisons)
    DMS = range(NB_DM)

    # Comparisons constraints
    preferences_changes = preferences_changes or [0] * NB_DM
    comparisons_refused = comparisons_refused or []

    # Alternatives
    alternatives = alternatives.subtable(
        list(set.union(*(set(comparisons[dm].elements) for dm in DMS)))  # type: ignore
    )

    # Criteria
    M = len(alternatives.criteria)  # type: ignore

    # Initial solutions
    match model:
        case ModelEnum.RMP:
            init_sols = [
                RMPModel.random(
                    nb_profiles=k,
                    nb_crit=M,
                    rng=rng,
                    profiles_values=midpoints(alternatives),
                )
                for rng in rng_(rng_init).spawn(nb_cpus)
            ]
        case ModelEnum.SRMP:
            init_sols = [
                SRMPModel.random(
                    nb_profiles=k,
                    nb_crit=M,
                    rng=rng,
                    profiles_values=midpoints(alternatives),
                )
                for rng in rng_(rng_init).spawn(nb_cpus)
            ]
        case _:
            raise ValueError(f"{model} model not compatible")
    if lex_order:
        for init_sol in init_sols:
            init_sol.lexicographic_order = lex_order

    # Neighborhood operators
    neighbors: list[Neighbor[SRMPModel | RMPModel]] = []
    prob: list[int] = []

    neighbors.append(
        NeighborProfileDiscretized(NormalPerformanceTable(midpoints(alternatives).data))
    )
    prob.append(k * M)

    match model:
        case ModelEnum.RMP:
            neighbors.append(NeighborImportanceRelation(False))
            prob.append(2**M)
        case ModelEnum.SRMP:
            if amp > 1:
                neighbors.append(NeighborWeight())
            else:
                neighbors.append(NeighborWeightAmp(amp))
            prob.append(M)

    if (not lex_order) and (k >= 2):
        neighbors.append(NeighborLexOrder(False))
        prob.append(k)

    neighbor = RandomNeighbor(neighbors, prob)

    # Objective
    objective = (
        FitnessObjective(alternatives, comparisons[0])
        if NB_DM == 1
        else CollectiveObjective(
            alternatives, comparisons, preferences_changes, comparisons_refused
        )
    )

    # Cooling schedule
    cooling_schedule = GeometricSchedule(alpha)

    # Random walk
    if t0 is None:
        assert accept
        t0 = initial_temperature(
            accept,
            neighbor,
            objective,
            init_sols[0],
            rng_sa,
            max(max_time // 100, 1) if max_time else None,
            max(max_it // 100, 1) if max_it else None,
        )

    # Simulated Annealing
    return (
        [
            SimulatedAnnealing(
                neighbor,
                objective,
                init_sol,
                rng,
                max_time,
                max_it,
                max_it_non_improving,
                log_file,
                t0,
                L,
                cooling_schedule,
                Tf,
            )
            for (init_sol, rng) in zip(init_sols, rng_(rng_sa).spawn(len(init_sols)))
        ],
        SenseEnum.MIN,
    )


def sa_result[S: Model](sa: SimulatedAnnealing[S]):
    sa.learn()

    return SAResult(sa.best_sol, sa.best_obj, sa.time, sa.it)
