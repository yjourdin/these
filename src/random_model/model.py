from dataclasses import dataclass

import numpy as np

from src.dataclass import RandomDataclass
from src.model import Group, Model
from src.performance_table.type import PerformanceTableType
from src.preference_structure.generate import random_preference_relation
from src.random import SeedMixin

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
