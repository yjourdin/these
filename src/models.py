from collections.abc import Container
from typing import cast

from .dataclass import Dataclass
from .enum import Enum, StrEnum
from .model import Model
from .random_model.model import RandomGroup, RandomModel
from .rmp.model import RMPParamEnum, rmp_group_model, rmp_model, rmp_model_from_name
from .srmp.model import (
    SRMPParamEnum,
    srmp_group_model,
    srmp_model,
    srmp_model_from_name,
)


class ModelEnum(StrEnum):
    RMP = "RMP"
    SRMP = "SRMP"
    RANDOM = "Random"


class GroupModelEnum(Enum):
    RMP_WPL = (
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_WP = (
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
            )
        ),
    )
    RMP_WL = (
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_PL = (
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_W = (
        ModelEnum.RMP,
        set((SRMPParamEnum.WEIGHTS,)),
    )
    RMP_P = (
        ModelEnum.RMP,
        set((SRMPParamEnum.PROFILES,)),
    )
    RMP_L = (
        ModelEnum.RMP,
        set((SRMPParamEnum.LEXICOGRAPHIC_ORDER,)),
    )
    RMP = (
        ModelEnum.RMP,
        set(),
    )  # type: ignore
    SRMP_WPL = (
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_WP = (
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
            )
        ),
    )
    SRMP_WL = (
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_PL = (
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_W = (
        ModelEnum.SRMP,
        set((SRMPParamEnum.WEIGHTS,)),
    )
    SRMP_P = (
        ModelEnum.SRMP,
        set((SRMPParamEnum.PROFILES,)),
    )
    SRMP_L = (
        ModelEnum.SRMP,
        set((SRMPParamEnum.LEXICOGRAPHIC_ORDER,)),
    )
    SRMP = (
        ModelEnum.SRMP,
        set(),
    )  # type: ignore
    RANDOM = (ModelEnum.RANDOM, set())  # type: ignore

    def __str__(self) -> str:
        return self.name


def group_model(
    model: ModelEnum, shared_params: Container[RMPParamEnum | SRMPParamEnum]
):
    if model is ModelEnum.RMP:
        shared_params = cast(Container[RMPParamEnum], shared_params)
        return rmp_group_model(shared_params)
    elif model is ModelEnum.SRMP:
        shared_params = cast(Container[SRMPParamEnum], shared_params)
        return srmp_group_model(shared_params)
    else:
        return RandomGroup


def model(
    model: ModelEnum,
    group_size: int,
    shared_params: Container[RMPParamEnum | SRMPParamEnum],
):
    if model is ModelEnum.RMP:
        shared_params = cast(Container[RMPParamEnum], shared_params)
        return rmp_model(group_size, shared_params)
    elif model is ModelEnum.SRMP:
        shared_params = cast(Container[SRMPParamEnum], shared_params)
        return srmp_model(group_size, shared_params)
    else:
        return RandomModel()


def model_from_json(s) -> Model:
    dct = Dataclass.json_to_dict(s)
    if not dct:
        raise ValueError("Empty json")
    classname = Dataclass.pop_class_name(dct)
    upper_classname = classname.upper()
    if ModelEnum.SRMP.value in upper_classname:
        cls = srmp_model_from_name(classname)
    elif ModelEnum.RMP.value in upper_classname:
        cls = rmp_model_from_name(classname)
    else:
        cls = RandomModel
    dct = cls.decode(dct)
    return cls.from_dict(dct)
