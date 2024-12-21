import math
from collections import defaultdict, deque
from typing import NamedTuple

from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure

from src.preference_structure.fitness import fitness_comparisons_ranking

from ..dataclass import Dataclass
from ..model import Model
from .neighborhood import Neighborhood


class ModelGraphResult[S](NamedTuple):
    parents: dict[S, list[S]]
    rankings: dict[S, Ranking]
    dm_models: list[list[S]]


class ModelGraph[S: Model](Dataclass):
    neighborhood: Neighborhood[S]

    def explore(
        self,
        source: S,
        targets: list[PreferenceStructure],
        alternatives: PerformanceTable,
    ):
        Q = deque([source])
        parents: defaultdict[S, list[S]] = defaultdict(list)
        rankings: dict[S, Ranking] = {source: source.rank(alternatives)}
        distances: dict[S, int] = {source: 0}
        dm_models: defaultdict[int, list[S]] = defaultdict(list)
        distances_max: list[float] = [math.inf] * len(targets)

        while Q:
            v = Q.popleft()
            rankings[v] = v.rank(alternatives)
            for dm, target in enumerate(targets):
                if distances[v] <= distances_max[dm]:
                    if fitness_comparisons_ranking(target, rankings[v]):
                        dm_models[dm].append(v)
                        distances_max[dm] = distances[v]
            if distances[v] < max(distances_max):
                for w in self.neighborhood(v):
                    if w not in rankings:
                        rankings[w] = w.rank(alternatives)
                        distances[w] = distances[v] + 1
                        Q.append(w)
                    if distances[w] == distances[v] + 1:
                        parents[w].append(v)

        return ModelGraphResult(
            parents, rankings, list(dm_models[dm] for dm in range(len(targets)))
        )
