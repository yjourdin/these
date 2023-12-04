from collections.abc import Collection, Iterator
from itertools import chain, combinations
from typing import Any, MutableMapping


class PowerSet(MutableMapping[frozenset[Any], int]):
    def __init__(self, s: Collection[Any]) -> None:
        s = frozenset(s)

        power_set_tmp = chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1)
        )
        power_set = frozenset(frozenset(i for i in ss) for ss in power_set_tmp)

        self.capacities = {ss: len(ss) for ss in power_set}
        self.supremum = {ss: {ss | {i} for i in (s - ss)} for ss in power_set}
        self.infimum = {ss: {ss - {i} for i in ss} for ss in power_set}

    def __getitem__(self, __key: frozenset[Any]) -> int:
        return self.capacities.__getitem__(__key)

    def __setitem__(self, __key: frozenset[Any], __value: int) -> None:
        if 0 <= __value < len(self.capacities):
            if self.min_capacity(__key) <= __value <= self.max_capacity(__key):
                return self.capacities.__setitem__(__key, __value)
            else:
                raise ValueError("Violates the monotonicity constraint")
        else:
            raise ValueError("Capacity must be between 0 and N-1")

    def __delitem__(self, __key: frozenset[Any]) -> None:
        return self.capacities.__delitem__(__key)

    def __iter__(self) -> Iterator[frozenset[Any]]:
        return self.capacities.__iter__()

    def __len__(self) -> int:
        return self.capacities.__len__()

    def min_capacity(self, __key: frozenset[Any]) -> int:
        infimum_capacities = [self.capacities[ss] for ss in self.infimum[__key]]
        return max(infimum_capacities) if infimum_capacities else 0

    def max_capacity(self, __key: frozenset[Any]) -> int:
        if len(__key) == 0:
            return 0
        supremum_capacities = [self.capacities[ss] for ss in self.supremum[__key]]
        return min(supremum_capacities) if supremum_capacities else len(self) - 1
