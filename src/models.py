from enum import Enum, auto

from .dataclass import Dataclass
from .model import Model, ParamFlag
from .random_model.model import RandomGroup, RandomModel
from .rmp.model import RMPParamFlag, rmp_group_model, rmp_model, rmp_model_from_name
from .srmp.model import (
    SRMPParamFlag,
    srmp_group_model,
    srmp_model,
    srmp_model_from_name,
)
from .strenum import StrEnumCustom


class ModelEnum(StrEnumCustom):
    RMP = auto()
    SRMP = auto()
    RANDOM = auto()


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
    RMP = (ModelEnum.RMP, RMPParamFlag(0))
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
    SRMP = (ModelEnum.SRMP, SRMPParamFlag(0))
    RANDOM = (ModelEnum.RANDOM, RMPParamFlag(0))

    def __init__(self, model: ModelEnum, shared_params: ParamFlag):
        self.model = model
        self.shared_params = shared_params

    def __str__(self) -> str:
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


def model_from_json(s: str) -> Model:
    dct = Dataclass.json_to_dict(s)
    if not dct:
        raise ValueError("Empty json")
    classname = Dataclass.pop_class_name(dct)
    lower_classname = classname.lower()
    if ModelEnum.SRMP in lower_classname:
        cls = srmp_model_from_name(classname)
    elif ModelEnum.RMP in lower_classname:
        cls = rmp_model_from_name(classname)
    else:
        cls = RandomModel
    dct = cls.decode(dct)
    return cls.from_dict(dct)
