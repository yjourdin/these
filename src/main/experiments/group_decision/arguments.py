from dataclasses import dataclass, field

from ....random import Seed
from ...arguments import Arguments, ExperimentEnum
from .seeds import Seeds


@dataclass
class ArgumentsGroupDecision(Arguments):
    experiment: ExperimentEnum = field(
        default=ExperimentEnum.GROUP_DECISION, init=False
    )
    seed: Seed | None = None
    nb_A_tr: int = 1
    nb_Mo: int | None = None
    nb_D: int | None = None
    seeds: Seeds = field(default_factory=Seeds)
    N_tr: list[int] = field(default_factory=list)
    group_size: list[int] = field(default_factory=lambda: [1])
    M: list[int] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    same_alt: list[bool] = field(default_factory=lambda: [True])
