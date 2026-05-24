from abc import ABC, abstractmethod

from mcda.relations import PreferenceStructure

from src.dataclass import Dataclass, dataclass, field
from src.model import Model
from src.performance_table.type import PerformanceTableType
from src.preference_structure.fitness import comparisons_ranking


class Objective[S](ABC):
    @abstractmethod
    def __call__(self, sol: S) -> float: ...

    @property
    @abstractmethod
    def optimum(self) -> float: ...


@dataclass
class FitnessObjective(Objective[Model], Dataclass):
    train_data: PerformanceTableType
    target: PreferenceStructure

    def __call__(self, sol: Model):
        return 1 - sol.fitness(self.train_data, self.target)

    @property
    def optimum(self):
        return 0


@dataclass
class CollectiveObjective(Objective[Model], Dataclass):
    performance_table: PerformanceTableType
    comparisons: list[PreferenceStructure]
    preferences_changes: list[int]
    comparisons_accepted: PreferenceStructure
    comparisons_refused: PreferenceStructure
    M: int = field(init=False)
    nb_DM: int = field(init=False)

    def __post_init__(self):
        self.M = max(len(comp) for comp in self.comparisons)  # type: ignore
        self.nb_DM = len(self.comparisons)

    def __call__(self, sol: Model):
        result = 0

        ranks = sol.rank_series(self.performance_table).to_dict()

        if not_accepted := comparisons_ranking(self.comparisons_accepted, ranks):
            result += len(not_accepted) * self.M**self.nb_DM

        if set(self.comparisons_refused) != set(
            refused := comparisons_ranking(self.comparisons_refused, ranks)
        ):
            result += (
                len(self.comparisons_refused) - len(refused)
            ) * self.M**self.nb_DM

        tup = sorted(
            self.preferences_changes[dm]
            + len(comparisons_ranking(self.comparisons[dm], ranks))
            for dm in range(self.nb_DM)
        )
        result += sum(x * self.M**i for (i, x) in enumerate(tup))

        return result

    @property
    def optimum(self):
        return 0
