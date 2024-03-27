from mcda.core.relations import PreferenceStructure
from numpy.random import Generator

from model import ModelType
from performance_table.normal_performance_table import NormalPerformanceTable
from rmp.generate import balanced_rmp
from srmp.generate import balanced_srmp
from utils import midpoints

from .cooling_schedule import GeometricSchedule
from .neighbors import (
    NeighborCapacities,
    NeighborLexOrder,
    NeighborProfiles,
    NeighborWeights,
    RandomNeighbor,
)
from .objective import FitnessObjective
from .sa import Neighbor, SimulatedAnnealing


def learn_sa(
    model: ModelType,
    k: int,
    alternatives: NormalPerformanceTable,
    comparisons: PreferenceStructure,
    T0: float,
    alpha: float,
    rng_init: Generator,
    rng_sa: Generator,
    L: int = 1,
    Tf: float | None = None,
    max_time: int | None = None,
    max_iter: int | None = None,
    max_iter_non_improving: int | None = None,
    verbose: bool = False,
):
    alternatives = alternatives.subtable(comparisons.elements)

    M = len(alternatives.criteria)

    neighbors: list[Neighbor] = []
    prob: list[int] = []

    neighbors.append(NeighborProfiles(midpoints(alternatives)))
    prob.append(k * M)

    match model:
        case "RMP":
            neighbors.append(NeighborCapacities(alternatives.criteria))
            prob.append(2**M)
        case "SRMP":
            neighbors.append(NeighborWeights(0.1))
            prob.append(M)

    if k >= 2:
        neighbors.append(NeighborLexOrder())
        prob.append(k)

    initial_model = None
    match model:
        case "RMP":
            initial_model = balanced_rmp(
                k,
                M,
                rng_init,
                midpoints(alternatives),
            )
        case "SRMP":
            initial_model = balanced_srmp(
                k,
                M,
                rng_init,
                midpoints(alternatives),
            )
    assert initial_model

    sa = SimulatedAnnealing(
        T0=T0,
        L=L,
        neighbor=RandomNeighbor(neighbors, prob),
        objective=FitnessObjective(alternatives, comparisons),
        cooling_schedule=GeometricSchedule(alpha),
        initial_sol=initial_model,
        rng=rng_sa,
        Tf=Tf,
        max_time=max_time,
        max_iter=max_iter,
        max_iter_non_improving=max_iter_non_improving,
        verbose=verbose,
    )

    sa.learn()

    return sa
