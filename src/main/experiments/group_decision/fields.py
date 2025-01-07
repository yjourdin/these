from dataclasses import dataclass, field

from ....field import Field, group_field, group_group_field
from ....field import field as custom_field
from ..elicitation.config import MIPConfig, create_config
from .hyperparameters import AcceptHyperparameters, GenHyperparameters


@custom_field("config")
@dataclass
class MIPConfigField(Field):
    config: MIPConfig

    @staticmethod
    def field_decode(o):
        return create_config(**o)


@group_field(fieldname="config", fieldclass=MIPConfigField)
@dataclass
class GroupMIPConfigField(Field):
    config: list[MIPConfig] = field(default_factory=list)


@custom_field("gen")
@dataclass
class GenHyperparameterField(Field):
    gen: GenHyperparameters

    @staticmethod
    def field_decode(o):
        return GenHyperparameters.from_dict(o)


@group_field(fieldname="gen", fieldclass=GenHyperparameterField)
@dataclass
class GroupGenHyperparameterField(Field):
    gen: list[GenHyperparameters] = field(default_factory=list)


@custom_field("accept")
@dataclass
class AcceptHyperparameterField(Field):
    accept: AcceptHyperparameters

    @staticmethod
    def field_decode(o):
        return AcceptHyperparameters.from_dict(o)


@group_group_field(fieldname="accept", fieldclass=AcceptHyperparameterField)
@dataclass
class GroupAcceptHyperparameterField(Field):
    accept: list[list[AcceptHyperparameters]] = field(default_factory=list)
