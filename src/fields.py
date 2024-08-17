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

    @classmethod
    def field_to_dict(cls, o):
        return MethodEnum(o)

    @classmethod
    def field_to_json(cls, o):
        return o.value


@group_field(fieldname="method", fieldclass=MethodField)
@dataclass
class GroupMethodField(Field):
    method: list[MethodEnum] = field(default_factory=list)


@dataclass
class ModelField(Field):
    Mo: GroupModelEnum

    @classmethod
    def field_to_dict(cls, o):
        return GroupModelEnum[o]

    @classmethod
    def field_to_json(cls, o):
        return o.name


@group_field(fieldname="Mo", fieldclass=ModelField)
@dataclass
class GroupMoField(Field):
    Mo: list[GroupModelEnum] = field(default_factory=list)


@group_field(fieldname="Me", fieldclass=ModelField)
@dataclass
class GroupMeField(Field):
    Me: list[GroupModelEnum] | None = None
