from enum import Enum
from typing import Any

from .dataclass import Dataclass
from .model import ParamFlag
from .random_model.model import RandomGroup, RandomModel
from .rmp.model import (
    RMPModel,
    RMPParamFlag,
    rmp_group_model,
    rmp_model,
    rmp_model_from_name,
)
from .srmp.model import (
    SRMPModel,
    SRMPParamFlag,
    srmp_group_model,
    srmp_model,
    srmp_model_from_name,
)


class ModelEnum(Enum):
    RMP = RMPModel
    SRMP = SRMPModel
    RANDOM = RandomModel

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            value = value.upper()
            for member in cls:
                if member.name == value:
                    return member
        return None

    def __str__(self):
        return self.name


class GroupModelEnum(Enum):
    RMP_IPL = (
        ModelEnum.RMP,
        RMPParamFlag.IMPORTANCE_RELATION
        | RMPParamFlag.PROFILES
        | RMPParamFlag.LEXICOGRAPHIC_ORDER,
    )
    RMP_IP = (ModelEnum.RMP, RMPParamFlag.IMPORTANCE_RELATION | RMPParamFlag.PROFILES)
    RMP_IL = (
        ModelEnum.RMP,
        RMPParamFlag.IMPORTANCE_RELATION | RMPParamFlag.LEXICOGRAPHIC_ORDER,
    )
    RMP_PL = (ModelEnum.RMP, RMPParamFlag.PROFILES | RMPParamFlag.LEXICOGRAPHIC_ORDER)
    RMP_I = (ModelEnum.RMP, RMPParamFlag.IMPORTANCE_RELATION)
    RMP_P = (ModelEnum.RMP, RMPParamFlag.PROFILES)
    RMP_L = (ModelEnum.RMP, RMPParamFlag.LEXICOGRAPHIC_ORDER)
    RMP = (ModelEnum.RMP, RMPParamFlag.NONE)
    SRMP_WPL = (
        ModelEnum.SRMP,
        SRMPParamFlag.WEIGHTS
        | SRMPParamFlag.PROFILES
        | SRMPParamFlag.LEXICOGRAPHIC_ORDER,
    )
    SRMP_WP = (ModelEnum.SRMP, SRMPParamFlag.WEIGHTS | SRMPParamFlag.PROFILES)
    SRMP_WL = (
        ModelEnum.SRMP,
        SRMPParamFlag.WEIGHTS | SRMPParamFlag.LEXICOGRAPHIC_ORDER,
    )
    SRMP_PL = (
        ModelEnum.SRMP,
        SRMPParamFlag.PROFILES | SRMPParamFlag.LEXICOGRAPHIC_ORDER,
    )
    SRMP_W = (ModelEnum.SRMP, SRMPParamFlag.WEIGHTS)
    SRMP_P = (ModelEnum.SRMP, SRMPParamFlag.PROFILES)
    SRMP_L = (ModelEnum.SRMP, SRMPParamFlag.LEXICOGRAPHIC_ORDER)
    SRMP = (ModelEnum.SRMP, SRMPParamFlag.NONE)
    RANDOM = (ModelEnum.RANDOM, RMPParamFlag.NONE)

    def __init__(self, model: ModelEnum, shared_params: ParamFlag):
        self.model = model
        self.shared_params = shared_params

    def __str__(self):
        return self.name


def group_model(model: ModelEnum, shared_params: ParamFlag):
    if model is ModelEnum.RMP:
        shared_params = RMPParamFlag(shared_params)
        return rmp_group_model(shared_params)
    elif model is ModelEnum.SRMP:
        shared_params = SRMPParamFlag(shared_params)
        return srmp_group_model(shared_params)
    else:
        return RandomGroup


def model(
    group_model: GroupModelEnum,
    group_size: int,
):
    match group_model.model:
        case ModelEnum.RMP:
            return rmp_model(group_size, RMPParamFlag(group_model.shared_params))
        case ModelEnum.SRMP:
            return srmp_model(group_size, SRMPParamFlag(group_model.shared_params))
        case ModelEnum.RANDOM:
            return RandomModel


def model_from_json(s: str):
    dct = Dataclass.json_to_dict(s)
    if not dct:
        raise ValueError("Empty json")
    classname = Dataclass.pop_class_name(dct)
    upper_classname = classname.upper()
    if "SRMP" in upper_classname:
        cls = srmp_model_from_name(classname)
    elif "RMP" in upper_classname:
        cls = rmp_model_from_name(classname)
    else:
        cls = RandomModel
    dct = cls.decode(dct)
    return cls.from_dict(dct)
