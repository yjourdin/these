from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Self, overload

from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure
from numpy.random import Generator

from src.utils import list_replace

from .aggregator import agg_float, agg_rank
from .dataclass import RandomDataclass, RandomFrozenDataclass
from .preference_structure.fitness import fitness_comparisons_ranking
from .random import Random


@dataclass
class Model(RandomDataclass):
    @abstractmethod
    def rank(self, performance_table: PerformanceTable) -> Ranking: ...

    def fitness(
        self, performance_table: PerformanceTable, comparisons: PreferenceStructure
    ):
        return fitness_comparisons_ranking(comparisons, self.rank(performance_table))

    @classmethod
    def from_reference(cls, other: Self, rng: Generator, *args, **kwargs):
        return deepcopy(other)


@dataclass(frozen=True)
class FrozenModel[M: Model](RandomFrozenDataclass):
    @property
    @abstractmethod
    def model(self) -> M: ...


@dataclass
class AggModel(Model):
    models: list[Model]

    def rank(self, performance_table: PerformanceTable) -> Ranking:
        return agg_rank(model.rank(performance_table) for model in self.models)


@dataclass
class GroupModel[M: Model](Model, Sequence[M]):
    group_size: int
    dm_weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.dm_weights = list_replace([1] * self.group_size, self.dm_weights)

    @property
    def collective_model(self) -> Model:
        return AggModel(list(self))

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> M: ...

    @overload
    @abstractmethod
    def __getitem__(self, i: slice) -> Sequence[M]: ...

    @abstractmethod
    def __getitem__(self, i) -> M | Sequence[M]: ...

    def __len__(self):
        return self.group_size

    def rank(self, performance_table):
        return self.collective_model.rank(performance_table)

    def fitness(
        self,
        performance_table: PerformanceTable,
        comparisons: PreferenceStructure | list[PreferenceStructure],
    ):
        match comparisons:
            case PreferenceStructure():
                return super().fitness(performance_table, comparisons)
            case _:
                return agg_float(
                    map(
                        lambda x: x[0].fitness(performance_table, x[1]),
                        zip(self, comparisons),
                    )
                )


class Group[M: Model](list[M], GroupModel[M], Random):
    model: ClassVar[type[M]]  # type: ignore
    dm_models: list[M]

    def __getitem__(self, i):
        return self.dm_models[i]

    @property
    def group_size(self):
        return len(self.dm_models)

    def __str__(self):
        return "[" + ", ".join([str(model) for model in self]) + "]"

    @classmethod
    def random(cls, *args, **kwargs):
        return cls([cls.model.random(*args, **kwargs)])

    @classmethod
    def from_reference(cls, other: M, rng: Generator, *args, **kwargs):
        return cls([cls.model.from_reference(other, rng, *args, **kwargs)])
