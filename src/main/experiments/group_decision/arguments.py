from dataclasses import dataclass, field

from ....random import Seed
from ...arguments import Arguments, ExperimentEnum
from .fields import (
    GroupAcceptHyperparameterField,
    GroupGenHyperparameterField,
    GroupMIPConfigField,
)
from .seeds import Seeds


@dataclass
class ArgumentsGroupDecision(
    Arguments,
    GroupMIPConfigField,
    GroupGenHyperparameterField,
    GroupAcceptHyperparameterField,
):
    experiment: ExperimentEnum = field(
        default=ExperimentEnum.GROUP_DECISION, init=False
    )
    seed: Seed | None = None
    nb_Atr: int = 1
    nb_Mo: int | None = None
    nb_Mi: int | None = None
    group_size: list[int] = field(default_factory=list)
    nb_D: int | None = None
    nb_Mc: int | None = None
    seeds: Seeds = field(default_factory=Seeds)
    N_tr: list[int] = field(default_factory=list)
    M: list[int] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    same_alt: list[bool] = field(default_factory=lambda: [True])
    # gen: list[GenHyperparameters] = field(default_factory=list)
    # accept: list[list[AcceptHyperparameters]] = field(default_factory=list)
