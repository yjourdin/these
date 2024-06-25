from numpy.random import Generator

from .normal_performance_table import NormalPerformanceTable


def random_alternatives(
    nb_alt: int, nb_crit: int, rng: Generator
) -> NormalPerformanceTable:
    """Create a random alternatives dataset

    :param int nb_alt: Number of alternatives
    :param int nb_crit: Number of criteria
    :param Generator rng: Random generator
    :return PerformanceTable: Random alternatives dataset
    """
    return NormalPerformanceTable(rng.random((nb_alt, nb_crit)))
