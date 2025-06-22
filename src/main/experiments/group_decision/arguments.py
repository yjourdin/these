from dataclasses import dataclass, field

from ....constants import DEFAULT_MAX_TIME
from ...arguments import Arguments, ExperimentEnum
from .fields import GroupGroupParametersField, GroupMIPConfigField
from .seeds import Seeds


@dataclass
class ArgumentsGroupDecision(Arguments, GroupMIPConfigField, GroupGroupParametersField):
    experiment: ExperimentEnum = field(
        default=ExperimentEnum.GROUP_DECISION, init=False
    )
    max_time: int = DEFAULT_MAX_TIME
    seed: int | None = None
    nb_Atr: int = 1
    nb_Mo: int | None = None
    nb_Mi: int | None = None
    nb_D: int | None = None
    nb_Mie: int | None = None
    nb_Mc: int | None = None
    nb_P: int | None = None
    seeds: Seeds = field(default_factory=Seeds)
    group_size: list[int] = field(default_factory=list)
    N_tr: list[int] = field(default_factory=list)
    M: list[int] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    fixed_lex_order: bool = False
    Ke: list[int] | None = None
    same_alt: list[bool] = field(default_factory=lambda: [True])
    path: list[bool] = field(default_factory=lambda: [True])
    Mie: list[bool] = field(default_factory=lambda: [True])
