from dataclasses import dataclass, field

from ..field import Field
from ..field import field as custom_field
from ..field import group_field
from ..methods import MethodEnum
from ..models import GroupModelEnum
from .config import Config, create_config


@custom_field("method")
@dataclass
class MethodField(Field):
    method: MethodEnum

    @staticmethod
    def field_decode(o):
        return MethodEnum(o)


@group_field(fieldname="method", fieldclass=MethodField)
@dataclass
class GroupMethodField(Field):
    method: list[MethodEnum] = field(default_factory=list)


@dataclass
class ModelField(Field):
    Mo: GroupModelEnum

    @staticmethod
    def field_decode(o):
        return GroupModelEnum[o]


@group_field(fieldname="Mo", fieldclass=ModelField)
@dataclass
class GroupMoField(Field):
    Mo: list[GroupModelEnum] = field(default_factory=list)


@group_field(fieldname="Me", fieldclass=ModelField)
@dataclass
class GroupMeField(Field):
    Me: list[GroupModelEnum] | None = None


@custom_field("config")
@dataclass
class ConfigField(Field):
    config: Config

    @staticmethod
    def field_decode(o):
        return create_config(**o)


@group_field(fieldname="config", fieldclass=ConfigField)
@dataclass
class GroupConfigField(Field):
    config: list[Config] = field(default_factory=list)
