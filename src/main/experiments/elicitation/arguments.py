from dataclasses import dataclass, field

from ....random import Seed
from ...arguments import Arguments, ExperimentEnum
from .fields import GroupConfigField, GroupMeField, GroupMethodField, GroupMoField
from .seeds import Seeds


@dataclass
class ArgumentsElicitation(
    Arguments, GroupMethodField, GroupMoField, GroupMeField, GroupConfigField
):
    experiment: ExperimentEnum = field(default=ExperimentEnum.ELICITATION, init=False)
    seed: Seed | None = None
    nb_Atr: int = 1
    nb_Mo: int | None = None
    nb_Ate: int | None = None
    nb_D: int | None = None
    nb_Me: int | None = None
    seeds: Seeds = field(default_factory=Seeds)
    N_tr: list[int] = field(default_factory=list)
    N_te: list[int] | None = None
    group_size: list[int] = field(default_factory=lambda: [1])
    M: list[int] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    same_alt: list[bool] = field(default_factory=lambda: [True])
    fixed_lex_order: bool = False
    Ke: list[int] | None = None
    error: list[float] = field(default_factory=lambda: [0])
