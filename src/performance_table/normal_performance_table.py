from mcda.internal.core.scales import NormalScale, QuantitativeScale
from mcda.matrices import PerformanceTable


class NormalPerformanceTable(PerformanceTable[NormalScale]):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, scales=QuantitativeScale.normal(), **kwargs)
