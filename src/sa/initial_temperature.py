import io
from math import log

import pandas as pd
from numpy import diff, mean
from numpy.random import Generator

from .neighbor import Neighbor
from .objective import Objective
from .random_walk import RandomWalk
from .type import T


def initial_temperature(
    acceptance_rate: float,
    neighbor: Neighbor[T],
    objective: Objective,
    init_sol: T,
    rng: Generator,
    max_time: int | None = None,
    max_it: int | None = None,
):
    results = io.StringIO()

    RandomWalk(neighbor, objective, init_sol, rng, max_time, max_it, results).learn()

    results.seek(0)

    energy = -pd.read_csv(results, dialect="unix")["Obj"]
    transitions = diff(energy)
    positive_transitions = transitions[transitions >= 0]
    positive_transitions_mean = mean(positive_transitions)
    return float(positive_transitions_mean / (-log(acceptance_rate)))
