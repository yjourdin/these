from dataclasses import dataclass

from mcda import PerformanceTable

from ..dataclass import GeneratedDataclass
from ..model import Group, Model
from ..preference_structure.random_weak_order import random_preference_relation
from ..random import SeedMixin
from .field import SeedField


@dataclass
class RandomModel(Model, GeneratedDataclass, SeedField, SeedMixin):
    def rank(self, performance_table: PerformanceTable):
        return random_preference_relation(performance_table, self.rng)


class RandomGroup(Group[RandomModel]):
    model = RandomModel