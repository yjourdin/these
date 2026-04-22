from dataclasses import dataclass, field
from itertools import count
from typing import Any

from src.dataclass import FrozenDataclass
from src.field import Field, group_field
from src.field import field as custom_field

from ..elicitation.config import MIPConfig, create_config
from .seeds import Seeds


@dataclass(frozen=True)
class SRMPParametersDeviation(FrozenDataclass):
    P: float
    W: float
    L: int


@custom_field("gen")
@dataclass(frozen=True)
class GenField(Field):
    gen: SRMPParametersDeviation

    @staticmethod
    def field_decode(o: Any):
        return SRMPParametersDeviation.from_dict(o)


@custom_field("accept")
@dataclass(frozen=True)
class AcceptField(Field):
    accept: SRMPParametersDeviation

    @staticmethod
    def field_decode(o: Any):
        return SRMPParametersDeviation.from_dict(o)


@dataclass(frozen=True)
class GroupParameters(GenField, AcceptField, FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False, hash=False)

    def __str__(self) -> str:
        return str(self.id)


@custom_field("group")
@dataclass
class GroupParametersField(Field):
    group: GroupParameters

    @staticmethod
    def field_decode(o: Any):
        return GroupParameters(**GroupParameters.decode(o))


@group_field(fieldname="group", fieldclass=GroupParametersField)
@dataclass
class GroupGroupParametersField(Field):
    group: list[GroupParameters] = field(default_factory=list)


@custom_field("Mie_config")
@dataclass
class MieConfigField(Field):
    Mie_config: MIPConfig

    @staticmethod
    def field_decode(o: Any):
        return create_config(**o)


@group_field(fieldname="Mie_config", fieldclass=MieConfigField)
@dataclass
class GroupMieConfigField(Field):
    Mie_config: list[MIPConfig] = field(default_factory=list)


@custom_field("seeds")
@dataclass
class SeedsField(Field):
    seeds: Seeds = field(default_factory=Seeds)

    @staticmethod
    def field_decode(o: Any):
        return Seeds(**o)
