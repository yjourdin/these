from collections.abc import Container
from enum import Enum

from .rmp.model import (
    RMPGroupModel,
    RMPGroupModelImportance,
    RMPGroupModelImportanceLexicographic,
    RMPGroupModelImportanceProfiles,
    RMPGroupModelImportanceProfilesLexicographic,
    RMPGroupModelLexicographic,
    RMPGroupModelProfiles,
    RMPGroupModelProfilesLexicographic,
    RMPParamEnum,
)
from .srmp.model import (
    SRMPGroupModel,
    SRMPGroupModelLexicographic,
    SRMPGroupModelProfiles,
    SRMPGroupModelProfilesLexicographic,
    SRMPGroupModelWeights,
    SRMPGroupModelWeightsLexicographic,
    SRMPGroupModelWeightsProfiles,
    SRMPGroupModelWeightsProfilesLexicographic,
    SRMPParamEnum,
)


class ModelEnum(str, Enum):
    RMP = "RMP"
    SRMP = "SRMP"


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


def group_model(
    model: ModelEnum, shared_params: Container[RMPParamEnum | SRMPParamEnum]
):
    if model == ModelEnum.RMP:
        if RMPParamEnum.PROFILES in shared_params:
            if RMPParamEnum.IMPORTANCE_RELATION in shared_params:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return RMPGroupModelImportanceProfilesLexicographic
                else:
                    return RMPGroupModelImportanceProfiles
            else:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return RMPGroupModelProfilesLexicographic
                else:
                    return RMPGroupModelProfiles
        else:
            if RMPParamEnum.IMPORTANCE_RELATION in shared_params:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return RMPGroupModelImportanceLexicographic
                else:
                    return RMPGroupModelImportance
            else:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return RMPGroupModelLexicographic
                else:
                    return RMPGroupModel
    elif model == ModelEnum.SRMP:
        if SRMPParamEnum.PROFILES in shared_params:
            if SRMPParamEnum.WEIGHTS in shared_params:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return SRMPGroupModelWeightsProfilesLexicographic
                else:
                    return SRMPGroupModelWeightsProfiles
            else:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return SRMPGroupModelProfilesLexicographic
                else:
                    return SRMPGroupModelProfiles
        else:
            if SRMPParamEnum.WEIGHTS in shared_params:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return SRMPGroupModelWeightsLexicographic
                else:
                    return SRMPGroupModelWeights
            else:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in shared_params:
                    return SRMPGroupModelLexicographic
                else:
                    return SRMPGroupModel
