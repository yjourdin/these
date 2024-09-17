from mcda import PerformanceTable
from mcda.internal.core.scales import NormalScale, QuantitativeScale
from mcda.outranking.srmp import SRMP
from numpy.random import Generator


class NormalPerformanceTable(PerformanceTable[NormalScale]):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, scales=QuantitativeScale.normal(), **kwargs)

    @classmethod
    def random(cls, nb_alt: int, nb_crit: int, rng: Generator):
        return cls(rng.random((nb_alt, nb_crit)))

    def plot(self, *args, **kwargs):
        return SRMP.plot_input_data(self, *args, **kwargs)
