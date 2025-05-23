from dataclasses import dataclass

import numpy as np

from ..dataclass import RandomDataclass
from ..model import Group, Model
from ..performance_table.type import PerformanceTableType
from ..preference_structure.generate import random_preference_relation
from ..random import SeedMixin
from .field import SeedField


@dataclass
class RandomModel(Model, RandomDataclass, SeedField, SeedMixin):
    def rank_numpy(self, performance_table: PerformanceTableType):
        return (
            random_preference_relation(performance_table, self.rng())
            .ranking.data.to_numpy()
            .astype(np.int_)
        )


class RandomGroup(Group[RandomModel]):
    model = RandomModel
