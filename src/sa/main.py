from typing import NamedTuple, TextIO

from mcda.relations import PreferenceStructure

from ..constants import DEFAULT_MAX_TIME
from ..model import Model
from ..models import ModelEnum
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..random import RNGParam, rng_
from ..rmp.model import RMPModel
from ..srmp.model import SRMPModel
from ..utils import midpoints
from .cooling_schedule import GeometricSchedule
from .initial_temperature import initial_temperature
from .neighbor import (
    NeighborImportanceRelation,
    NeighborLexOrder,
    NeighborProfileDiscretized,
    NeighborWeight,
    NeighborWeightAmp,
    RandomNeighbor,
)
from .objective import FitnessObjective
from .sa import Neighbor, SimulatedAnnealing


class SAResult(NamedTuple):
    best_model: Model
    best_fitness: float
    time: float
    it: int


def learn_sa(
    model: ModelEnum,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: PreferenceStructure,
    alpha: float,
    amp: float,
    rng_init: RNGParam = None,
    rng_sa: RNGParam = None,
    lex_order: list[int] | None = None,
    t0: float | None = None,
    accept: float | None = None,
    L: int = 1,
    Tf: float | None = None,
    max_time: int = DEFAULT_MAX_TIME,
    max_it: int | None = None,
    max_it_non_improving: int | None = None,
    log_file: TextIO | None = None,
):
    alternatives = alternatives.subtable(comparisons.elements)

    M = len(alternatives.criteria)  # type: ignore

    match model.lower():
        case ModelEnum.RMP:
            init_sol = RMPModel.random(
                nb_profiles=k,
                nb_crit=M,
                rng=rng_init,
                profiles_values=midpoints(alternatives),
            )
        case ModelEnum.SRMP:
            init_sol = SRMPModel.random(
                nb_profiles=k,
                nb_crit=M,
                rng=rng_init,
                profiles_values=midpoints(alternatives),
            )
        case _:
            raise ValueError(f"{model} model not compatible")
    if lex_order:
        init_sol.lexicographic_order = lex_order

    neighbors: list[Neighbor[SRMPModel | RMPModel]] = []
    prob: list[int] = []

    neighbors.append(NeighborProfileDiscretized(midpoints(alternatives)))
    prob.append(k * M)

    match model.lower():
        case ModelEnum.RMP:
            neighbors.append(NeighborImportanceRelation(2**M - 1))
            prob.append(2**M)
        case ModelEnum.SRMP:
            if amp > 1:
                neighbors.append(NeighborWeight())  # type: ignore
            else:
                neighbors.append(NeighborWeightAmp(amp))  # type: ignore
            prob.append(M)
            pass
        case _:
            raise ValueError(f"{model} model not compatible")

    if (not lex_order) and (k >= 2):
        neighbors.append(NeighborLexOrder())
        prob.append(k)

    neighbor = RandomNeighbor(neighbors, prob)

    objective = FitnessObjective(alternatives, comparisons)

    cooling_schedule = GeometricSchedule(alpha)

    rng_sa = rng_(rng_sa)

    if t0 is None:
        assert accept
        t0 = initial_temperature(
            accept,
            neighbor,
            objective,
            init_sol,
            rng_sa,
            max(max_time // 100, 1) if max_time else None,
            max(max_it // 100, 1) if max_it else None,
        )

    sa = SimulatedAnnealing(
        t0,
        L,
        neighbor,
        objective,
        cooling_schedule,
        init_sol,
        rng_sa,
        Tf,
        max_time,
        max_it,
        max_it_non_improving,
        log_file,
    )

    sa.learn()

    return SAResult(sa.best_sol, 1 - sa.best_obj, sa.time, sa.it)
