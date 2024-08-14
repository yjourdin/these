from enum import Enum
from typing import NamedTuple

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


class GroupModelNamedTuple(NamedTuple):
    model: ModelEnum
    shared_params: set[RMPParamEnum | SRMPParamEnum]


class GroupModelEnum(GroupModelNamedTuple, Enum):
    RMP_WPL = GroupModelNamedTuple(
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_WP = GroupModelNamedTuple(
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
            )
        ),
    )
    RMP_WL = GroupModelNamedTuple(
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_PL = GroupModelNamedTuple(
        ModelEnum.RMP,
        set(
            (
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    RMP_W = GroupModelNamedTuple(
        ModelEnum.RMP,
        set((SRMPParamEnum.WEIGHTS,)),
    )
    RMP_P = GroupModelNamedTuple(
        ModelEnum.RMP,
        set((SRMPParamEnum.PROFILES,)),
    )
    RMP_L = GroupModelNamedTuple(
        ModelEnum.RMP,
        set((SRMPParamEnum.LEXICOGRAPHIC_ORDER,)),
    )
    RMP = GroupModelNamedTuple(
        ModelEnum.RMP,
        set(()),
    )
    SRMP_WPL = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_WP = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.PROFILES,
            )
        ),
    )
    SRMP_WL = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.WEIGHTS,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_PL = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set(
            (
                SRMPParamEnum.PROFILES,
                SRMPParamEnum.LEXICOGRAPHIC_ORDER,
            )
        ),
    )
    SRMP_W = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set((SRMPParamEnum.WEIGHTS,)),
    )
    SRMP_P = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set((SRMPParamEnum.PROFILES,)),
    )
    SRMP_L = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set((SRMPParamEnum.LEXICOGRAPHIC_ORDER,)),
    )
    SRMP = GroupModelNamedTuple(
        ModelEnum.SRMP,
        set(()),
    )


def group_model(group_model: GroupModelEnum):
    if group_model.model == ModelEnum.RMP:
        if RMPParamEnum.PROFILES in group_model.shared_params:
            if RMPParamEnum.IMPORTANCE_RELATION in group_model.shared_params:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return RMPGroupModelImportanceProfilesLexicographic
                else:
                    return RMPGroupModelImportanceProfiles
            else:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return RMPGroupModelProfilesLexicographic
                else:
                    return RMPGroupModelProfiles
        else:
            if RMPParamEnum.IMPORTANCE_RELATION in group_model.shared_params:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return RMPGroupModelImportanceLexicographic
                else:
                    return RMPGroupModelImportance
            else:
                if RMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return RMPGroupModelLexicographic
                else:
                    return RMPGroupModel
    elif group_model.model == ModelEnum.SRMP:
        if SRMPParamEnum.PROFILES in group_model.shared_params:
            if SRMPParamEnum.WEIGHTS in group_model.shared_params:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return SRMPGroupModelWeightsProfilesLexicographic
                else:
                    return SRMPGroupModelWeightsProfiles
            else:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return SRMPGroupModelProfilesLexicographic
                else:
                    return SRMPGroupModelProfiles
        else:
            if SRMPParamEnum.WEIGHTS in group_model.shared_params:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return SRMPGroupModelWeightsLexicographic
                else:
                    return SRMPGroupModelWeights
            else:
                if SRMPParamEnum.LEXICOGRAPHIC_ORDER in group_model.shared_params:
                    return SRMPGroupModelLexicographic
                else:
                    return SRMPGroupModel
    else:
        raise ValueError(f"group model : {group_model} not recognised")
