from dataclasses import dataclass, field

from .field import Field, group_field
from .methods import MethodEnum
from .models import GroupModelEnum  # type: ignore


@dataclass
class MethodField(Field):
    method: MethodEnum

    @classmethod
    def json_to_dict(cls, dct: dict):
        super().json_to_dict(dct)
        if "method" in dct:
            dct["method"] = MethodEnum(dct["method"])
            return dct["method"]

    @classmethod
    def dict_to_json(cls, dct: dict):
        super().dict_to_json(dct)
        if "method" in dct:
            dct["method"] = dct["method"].value
            return dct["method"]


@group_field(fieldname="method", fieldclass=MethodField)
@dataclass
class GroupMethodField(Field):
    method: list[MethodEnum] = field(default_factory=list)


@dataclass
class GroupModelField(Field):
    Mo: GroupModelEnum
    Me: GroupModelEnum

    @classmethod
    def json_to_dict(cls, dct: dict):
        super().json_to_dict(dct)
        if "Mo" in dct:
            dct["Mo"] = GroupModelEnum[dct["Mo"]]
            return dct["Mo"]
        if "Me" in dct:
            dct["Me"] = GroupModelEnum[dct["Me"]]
            return dct["Me"]

    @classmethod
    def dict_to_json(cls, dct: dict):
        super().dict_to_json(dct)
        if "Mo" in dct:
            dct["Mo"] = dct["Mo"].name
            return dct["Mo"]
        if "Me" in dct:
            dct["Me"] = dct["Me"].name
            return dct["Me"]


@group_field(fieldname="Mo", fieldclass=GroupModelField)
@dataclass
class GroupMoField(Field):
    Mo: list[GroupModelEnum] = field(default_factory=list)


@group_field(fieldname="Me", fieldclass=GroupModelField)
@dataclass
class GroupMeField(Field):
    Me: list[GroupModelEnum] = field(default_factory=list)
