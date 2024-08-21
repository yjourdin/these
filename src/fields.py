from dataclasses import dataclass, field

from .field import Field
from .field import field as custom_field
from .field import group_field
from .methods import MethodEnum
from .models import GroupModelEnum


@custom_field("method")
@dataclass
class MethodField(Field):
    method: MethodEnum

    @staticmethod
    def field_decode(o):
        return MethodEnum(o)

    @staticmethod
    def field_encode(o):
        return str(o)


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

    @staticmethod
    def field_encode(o):
        return str(o)


@group_field(fieldname="Mo", fieldclass=ModelField)
@dataclass
class GroupMoField(Field):
    Mo: list[GroupModelEnum] = field(default_factory=list)


@group_field(fieldname="Me", fieldclass=ModelField)
@dataclass
class GroupMeField(Field):
    Me: list[GroupModelEnum] | None = None
