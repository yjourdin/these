import numpy as np
from numpy.random import Generator


def random_weights(nb_crit: int, rng: Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(nb_crit))
