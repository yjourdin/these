from dataclasses import dataclass, field
from itertools import count
from typing import Any

from src.dataclass import FrozenDataclass
from src.field import Field, group_field
from src.field import field as custom_field

from ....models import ModelEnum
from ....utils import CustomException
from ..elicitation.config import MIPConfig, create_config
from .seeds import Seeds


@dataclass(frozen=True)
class ParametersDeviation(FrozenDataclass):
    ...


@dataclass(frozen=True)
class RMPParametersDeviation(ParametersDeviation):
    P: float
    I: int
    L: int

@dataclass(frozen=True)
class SRMPParametersDeviation(ParametersDeviation):
    P: float
    W: float
    L: int

def parameters_deviation_from_dict(o: Any):
    if "I" in o:
        return RMPParametersDeviation.from_dict(o)
    elif "W" in o:
        return SRMPParametersDeviation.from_dict(o)
    else:
        raise CustomException("Unknown parameters deviation")


@custom_field("gen")
@dataclass(frozen=True)
class GenField[T: ParametersDeviation](Field):
    gen: T

    @staticmethod
    def field_decode(o: Any):
        return parameters_deviation_from_dict(0)


@custom_field("accept")
@dataclass(frozen=True)
class AcceptField[T: ParametersDeviation](Field):
    accept: T

    @staticmethod
    def field_decode(o: Any):
        return parameters_deviation_from_dict(0)


@dataclass(frozen=True)
class GroupParameters[T: ParametersDeviation](GenField[T], AcceptField[T], FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False, hash=False)

    def __str__(self) -> str:
        return str(self.id)


GroupParametersT = GroupParameters[ParametersDeviation]


@custom_field("group")
@dataclass
class GroupParametersField(Field):
    group: GroupParametersT

    @staticmethod
    def field_decode(o: Any):
        return GroupParameters(**GroupParameters.decode(o))


@group_field(fieldname="group", fieldclass=GroupParametersField)
@dataclass
class GroupGroupParametersField(Field):
    group: list[GroupParametersT] = field(default_factory=list)


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


@dataclass
class ModelField(Field):
    model: ModelEnum

    @staticmethod
    def field_decode(o: Any):
        return ModelEnum[o]


@group_field(fieldname="model", fieldclass=ModelField)
@dataclass
class GroupModelField(Field):
    model: list[ModelEnum] = field(default_factory=list)