from collections.abc import Container
from dataclasses import dataclass

from mcda.internal.core.scales import NormalScale

from ..dataclass import GeneratedDataclass
from ..enum import StrEnum
from ..model import GroupModel, Model
from ..utils import print_list
from .field import (
    CapacityField,
    GroupImportanceRelationField,
    GroupLexicographicOrderField,
    GroupProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
    ProfilesField,
)
from .importance_relation import ImportanceRelation
from .rmp import NormalRMP


class RMPParamEnum(StrEnum):
    PROFILES = "profiles"
    IMPORTANCE_RELATION = "importance_relation"
    LEXICOGRAPHIC_ORDER = "lexicographic_order"


@dataclass
class RMPModel(  # type: ignore
    Model[NormalScale],
    GeneratedDataclass,
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
class RMPModelCapacity(  # type: ignore
    Model[NormalScale],
    GeneratedDataclass,
    ProfilesField,
    CapacityField,
    LexicographicOrderField,
):

    def __str__(self) -> str:
        return (
            f"{print_list(self.profiles.data.to_numpy()[0])}\t"
            f"{self.lexicographic_order.__str__()}"
        )

    def rank(self, performance_table):
        return NormalRMP(
            performance_table,
            ImportanceRelation.from_capacity(self.capacity),
            self.profiles,
            self.lexicographic_order,
        ).rank()


@dataclass
class RMPGroupModelImportanceProfilesLexicographic(  # type: ignore
    GroupModel[NormalScale],
    GeneratedDataclass,
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


@dataclass
class RMPGroupModelImportanceProfiles(  # type: ignore
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
) -> type[GroupModel[NormalScale]]:
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


def rmp_model(
    size: int, shared_params: Container[RMPParamEnum] = set()
) -> type[Model[NormalScale]]:
    if size == 1:
        return RMPModel
    else:
        return rmp_group_model(shared_params)


def rmp_model_from_name(name: str) -> type[Model[NormalScale]]:
    return eval(name)
