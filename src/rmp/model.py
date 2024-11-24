from collections.abc import Container
from dataclasses import dataclass

from ..dataclass import RandomDataclass
from ..enum_base import StrEnum
from ..model import GroupModel, Model
from ..utils import print_list
from .field import (
    GroupImportanceRelationField,
    GroupLexicographicOrderField,
    GroupProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
    ProfilesField,
)
from .rmp import NormalRMP


class RMPParamEnum(StrEnum):
    PROFILES = "profiles"
    IMPORTANCE_RELATION = "importance_relation"
    LEXICOGRAPHIC_ORDER = "lexicographic_order"


@dataclass
class RMPModel(  # type: ignore
    Model,
    RandomDataclass,
    ProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
):
    def __str__(self) -> str:
        return (
            print_list(self.profiles.data.to_numpy()[0])
            + "\t"
            + self.lexicographic_order.__str__()
        )

    def rank(self, performance_table):
        return NormalRMP(
            performance_table,
            self.importance_relation,
            self.profiles,
            self.lexicographic_order,
        ).rank()


@dataclass
class RMPGroupModelImportanceProfilesLexicographic(  # type: ignore
    GroupModel[RMPModel],
    RandomDataclass,
    ProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order,
        )

    @property
    def collective_model(self) -> Model:
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class RMPGroupModelImportanceProfiles(  # type: ignore
    GroupModel[RMPModel],
    RandomDataclass,
    ProfilesField,
    ImportanceRelationField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class RMPGroupModelImportanceLexicographic(  # type: ignore
    GroupModel[RMPModel],
    RandomDataclass,
    GroupProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles[i],
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class RMPGroupModelProfilesLexicographic(  # type: ignore
    GroupModel[RMPModel],
    RandomDataclass,
    ProfilesField,
    GroupImportanceRelationField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class RMPGroupModelImportance(
    GroupModel[RMPModel],
    RandomDataclass,
    GroupProfilesField,
    ImportanceRelationField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles[i],
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class RMPGroupModelProfiles(
    GroupModel[RMPModel],
    RandomDataclass,
    ProfilesField,
    GroupImportanceRelationField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation[i],
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class RMPGroupModelLexicographic(
    GroupModel[RMPModel],
    RandomDataclass,
    GroupProfilesField,
    GroupImportanceRelationField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles[i],
            importance_relation=self.importance_relation[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class RMPGroupModel(
    GroupModel[RMPModel],
    RandomDataclass,
    GroupProfilesField,
    GroupImportanceRelationField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return RMPModel(
            profiles=self.profiles[i],
            importance_relation=self.importance_relation[i],
            lexicographic_order=self.lexicographic_order[i],
        )


def rmp_group_model(
    shared_params: Container[RMPParamEnum],
) -> type[GroupModel[RMPModel]]:
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


def rmp_model(group_size: int, shared_params: Container[RMPParamEnum] = set()):
    if group_size == 1:
        return RMPModel
    else:
        return rmp_group_model(shared_params)


def rmp_model_from_name(name: str) -> type[Model]:
    return eval(name)
