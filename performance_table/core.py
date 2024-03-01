from mcda.core.matrices import PerformanceTable
from mcda.core.scales import NormalScale


class NormalPerformanceTable(PerformanceTable[NormalScale]):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, scales=NormalScale(), **kwargs)
