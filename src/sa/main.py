from typing import NamedTuple

from mcda.relations import PreferenceStructure
from numpy.random import Generator

from ..abstract_model import Model
from ..model import ModelType
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..rmp.generate import balanced_rmp
from ..srmp.generate import balanced_srmp
from ..utils import midpoints
from .cooling_schedule import GeometricSchedule
from .initial_temperature import initial_temperature
from .neighbor import (
    NeighborCapacities,
    NeighborLexOrder,
    NeighborProfiles,
    NeighborWeights,
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
    model: ModelType,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: PreferenceStructure,
    alpha: float,
    rng_init: Generator,
    rng_sa: Generator,
    T0: float | None = None,
    accept: float | None = None,
    L: int = 1,
    amp: float | None = None,
    Tf: float | None = None,
    max_time: int | None = None,
    max_it: int | None = None,
    max_it_non_improving: int | None = None,
    log_file=None,
):
    alternatives = alternatives.subtable(comparisons.elements)

    M = len(alternatives.criteria)

    init_model = None
    match model:
        case "RMP":
            init_model = balanced_rmp(
                k,
                M,
                rng_init,
                midpoints(alternatives),
            )
        case "SRMP":
            init_model = balanced_srmp(
                k,
                M,
                rng_init,
                midpoints(alternatives),
            )
    assert init_model

    neighbors: list[Neighbor] = []
    prob: list[int] = []

    neighbors.append(NeighborProfiles(midpoints(alternatives)))
    prob.append(k * M)

    match model:
        case "RMP":
            neighbors.append(NeighborCapacities(alternatives.criteria))
            prob.append(2**M)
        case "SRMP":
            if amp is not None:
                neighbors.append(NeighborWeights(amp))
            else:
                raise ValueError("amp must not be None")
            prob.append(M)

    if k >= 2:
        neighbors.append(NeighborLexOrder())
        prob.append(k)

    neighbor = RandomNeighbor(neighbors, prob)

    objective = FitnessObjective(alternatives, comparisons)

    cooling_schedule = GeometricSchedule(alpha)

    if T0 is None:
        assert accept
        T0 = initial_temperature(
            accept,
            neighbor,
            objective,
            init_model,
            rng_sa,
            max_time // 100 if max_time else None,
            max_it // 100 if max_it else None,
        )

    sa = SimulatedAnnealing(
        T0,
        L,
        neighbor,
        objective,
        cooling_schedule,
        init_model,
        rng_sa,
        Tf,
        max_time,
        max_it,
        max_it_non_improving,
        log_file,
    )

    sa.learn()

    return SAResult(sa.best_sol, 1 - sa.best_obj, sa.time, sa.it)
