import numpy as np
from numpy.random import Generator


def random_weights(nb_crit: int, rng: Generator) -> list[float]:
    return np.diff(
        np.sort(np.concatenate([[0], rng.random(nb_crit - 1), [1]]))
    ).tolist()


def balanced_weights(nb_crit: int):
    return [1 / nb_crit] * nb_crit
