from dataclasses import dataclass, field

from numpy.random import default_rng

from ..dataclass import Dataclass
from ..seed import seeds

Seed = int


@dataclass
class GroupSeed(Dataclass):
    group: Seed = -1
    dm: list[Seed] = field(default_factory=list)


def group_seed(seed: int, size: int):
    return GroupSeed(seed, seeds(default_rng(seed), size))


@dataclass
class Seeds(Dataclass):
    A_tr: list[Seed] = field(default_factory=list)
    A_te: list[Seed] = field(default_factory=list)
    Mo: dict[int, list[GroupSeed]] = field(default_factory=dict)
    D: list[Seed] = field(default_factory=list)
    Me: list[Seed] = field(default_factory=list)
