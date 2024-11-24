import numpy as np
from numpy.random import Generator


def random_weights(nb_crit: int, rng: Generator) -> list[float]:
    return np.diff(
        np.sort(np.concatenate([[0], rng.random(nb_crit - 1), [1]]))
    ).tolist()
