import io
from math import log

import numpy as np
import pandas as pd
from numpy.random import Generator

from .neighbor import Neighbor
from .objective import Objective
from .random_walk import RandomWalk


def initial_temperature[
    S
](
    acceptance_rate: float,
    neighbor: Neighbor[S],
    objective: Objective,
    init_sol: S,
    rng: Generator,
    max_time: int | None = None,
    max_it: int | None = None,
):
    results = io.StringIO()

    RandomWalk(neighbor, objective, init_sol, rng, max_time, max_it, results).learn()

    results.seek(0)

    energy = -pd.read_csv(results, dialect="unix")["Obj"]
    transitions = np.diff(energy)
    positive_transitions = transitions[transitions >= 0]
    positive_transitions_mean = np.mean(positive_transitions)
    return float(positive_transitions_mean / (-log(acceptance_rate)))
