from dataclasses import dataclass, field

from ..random import Seed

from ..dataclass import Dataclass


@dataclass
class Seeds(Dataclass):
    A_tr: list[Seed] = field(default_factory=list)
    A_te: list[Seed] = field(default_factory=list)
    Mo: dict[int, list[Seed]] = field(default_factory=dict)
    D: list[Seed] = field(default_factory=list)
    Me: list[Seed] = field(default_factory=list)
