from mcda.internal.core.scales import NormalScale, QuantitativeScale
from mcda.matrices import PerformanceTable
from numpy.random import Generator


class NormalPerformanceTable(PerformanceTable[NormalScale]):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, scales=QuantitativeScale.normal(), **kwargs)

    @classmethod
    def random(cls, nb_alt: int, nb_crit: int, rng: Generator):
        return cls(rng.random((nb_alt, nb_crit)))
