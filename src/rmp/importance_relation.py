import numpy as np

from ..julia.function import generate_weak_order_ext
from ..random import seed
from ..weak_order import WeakOrder


class ImportanceRelation(WeakOrder[frozenset[int]]):
    @classmethod
    def random(cls, nb_crit, rng):
        weak_order_ext = generate_weak_order_ext(nb_crit, seed(rng))

        labels = np.arange(nb_crit)
        dct: dict[frozenset[int], int] = {}

        for i, block in enumerate(weak_order_ext):
            for node in block:
                dct[frozenset(labels[node].tolist())] = i

        we = cls()
        we.dict = dct
        return we

    def max(self, key: frozenset[int]):
        try:
            return min(v for k, v in self.items() if key < k)
        except ValueError:
            return min(
                max({k: v for k, v in self.items() if k != key}.values()) + 1,
                len(self) - 1,
            )

    def min(self, key: frozenset[int]):
        try:
            return max(v for k, v in self.items() if k < key)
        except ValueError:
            return max(
                min({k: v for k, v in self.items() if k != key}.values()) - 1,
                0,
            )
