from abc import abstractmethod
from collections import UserList
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

from mcda.internal.core.scales import Scale
from mcda.internal.core.values import Ranking
from mcda import PerformanceTable
from mcda.relations import PreferenceStructure

from .aggregator import agg_float
from .dataclass import GeneratedDataclass
from .preference_structure.fitness import fitness_comparisons_ranking

S = TypeVar("S", bound=Scale, covariant=True)


@dataclass
class Model(GeneratedDataclass, Generic[S]):
    @abstractmethod
    def rank(self, performance_table: PerformanceTable[S]) -> Ranking: ...

    def fitness(
        self, performance_table: PerformanceTable[S], comparisons: PreferenceStructure
    ):
        return fitness_comparisons_ranking(comparisons, self.rank(performance_table))


@dataclass
class GroupModel(Generic[S], Model[S], Sequence[Model[S]]):
    size: int

    @overload
    def __getitem__(self, i: int) -> Model[S]: ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[Model[S]]: ...

    @abstractmethod
    def __getitem__(self, i): ...  # type: ignore

    def __len__(self):
        return self.size

    def rank(self, performance_table):
        return self[0].rank(performance_table)

    def fitness(
        self,
        performance_table: PerformanceTable[S],
        comparisons: PreferenceStructure | list[PreferenceStructure],
    ) -> float:
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


class Group(Generic[S], UserList, GroupModel[S]):
    @property
    def size(self):
        return len(self)

    def __str__(self) -> str:
        return "[" + ", ".join([str(model) for model in self]) + "]"
