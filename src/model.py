from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, SupportsIndex, overload

import numpy as np
import numpy.typing as npt
from mcda.internal.core.scales import DiscreteQuantitativeScale, PreferenceDirection
from mcda.internal.core.values import CommensurableValues, Ranking
from mcda.relations import PreferenceStructure
from numpy.random import Generator
from pandas import Series

from .aggregator import agg_float, agg_rank
from .dataclass import RandomDataclass, RandomFrozenDataclass
from .performance_table.type import PerformanceTableType
from .preference_structure.fitness import fitness_comparisons_ranking
from .random import Random
from .utils import list_replace


@dataclass
class Model(RandomDataclass):
    @abstractmethod
    def rank_numpy(
        self, performance_table: PerformanceTableType
    ) -> npt.NDArray[np.int_]: ...

    def rank_series(self, performance_table: PerformanceTableType):
        return Series(
            self.rank_numpy(performance_table),
            performance_table.alternatives,
            dtype=int,
        )

    def rank(self, performance_table: PerformanceTableType) -> Ranking:
        ranking = self.rank_series(performance_table)
        return CommensurableValues(
            ranking,
            scale=DiscreteQuantitativeScale(
                ranking.unique(),  # type: ignore
                PreferenceDirection.MIN,
            ),
        )

    def fitness(
        self, performance_table: PerformanceTableType, comparisons: PreferenceStructure
    ):
        return fitness_comparisons_ranking(
            comparisons, self.rank_series(performance_table)
        )

    @classmethod
    def from_reference(cls, other: Self, rng: Generator, *args: Any, **kwargs: Any):
        return deepcopy(other)


@dataclass(frozen=True)
class FrozenModel[M: Model](RandomFrozenDataclass):
    @property
    @abstractmethod
    def model(self) -> M: ...


@dataclass
class AggModel(Model):
    models: list[Model]

    def rank_numpy(self, performance_table: PerformanceTableType):
        return agg_rank(model.rank_numpy(performance_table) for model in self.models)


@dataclass
class GroupModel[M: Model](Model, Sequence[M]):
    _group_size: int
    dm_weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.dm_weights = list_replace([1] * self.group_size, self.dm_weights)

    @property
    def group_size(self):
        return self._group_size

    @property
    def collective_model(self) -> Model:
        return AggModel(list(self))

    @overload
    @abstractmethod
    def __getitem__(self, i: SupportsIndex) -> M: ...

    @overload
    @abstractmethod
    def __getitem__(self, i: slice) -> Sequence[M]: ...

    @abstractmethod
    def __getitem__(self, i: SupportsIndex | slice) -> M | Sequence[M]: ...

    def __len__(self):
        return self.group_size

    def rank_numpy(self, performance_table: PerformanceTableType):
        return self.collective_model.rank_numpy(performance_table)

    def fitness(
        self,
        performance_table: PerformanceTableType,
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


class Group[M: Model](list[M], GroupModel[M], Random):  # type: ignore
    model: ClassVar[type[M]]  # type: ignore
    dm_models: list[M]

    @overload
    def __getitem__(self, i: SupportsIndex) -> M: ...
    @overload
    def __getitem__(self, i: slice) -> list[M]: ...

    def __getitem__(self, i: SupportsIndex | slice) -> M | list[M]:
        return self.dm_models[i]

    @property
    def group_size(self):
        return len(self.dm_models)

    def __str__(self):
        return "[" + ", ".join([str(model) for model in self]) + "]"

    @classmethod
    def random(cls, *args: Any, **kwargs: Any):
        return cls([cls.model.random(*args, **kwargs)])

    @classmethod
    def from_reference(cls, other: M, rng: Generator, *args: Any, **kwargs: Any):
        return cls([cls.model.from_reference(other, rng, *args, **kwargs)])
