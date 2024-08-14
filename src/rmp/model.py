from dataclasses import dataclass
from enum import Enum

from mcda.internal.core.scales import NormalScale

from ..dataclass import GeneratedDataclass
from ..model import Model
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


class RMPParamEnum(Enum):
    PROFILES = "profiles"
    IMPORTANCE_RELATION = "importance_relation"
    LEXICOGRAPHIC_ORDER = "lexicographic_order"


@dataclass
class RMPModel(
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
class RMPModelCapacity(
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
class RMPGroupModelImportanceProfilesLexicographic(
    Model[NormalScale],
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
class RMPGroupModelImportanceProfiles(
    Model[NormalScale],
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
class RMPGroupModelImportanceLexicographic(
    Model[NormalScale],
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
class RMPGroupModelProfilesLexicographic(
    Model[NormalScale],
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
    Model[NormalScale],
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
    Model[NormalScale],
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
    Model[NormalScale],
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
    Model[NormalScale],
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