from dataclasses import dataclass
from enum import auto
from typing import Self

from numpy.random import Generator

from ..enum import ParamFlag
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
from .perturbations import PerturbImportanceRelation, PerturbLexOrder, PerturbProfile
from .rmp import NormalRMP


class RMPParamFlag(ParamFlag):
    PROFILES = auto()
    IMPORTANCE_RELATION = auto()
    LEXICOGRAPHIC_ORDER = auto()


@dataclass(unsafe_hash=True)
class RMPModel(
    Model,
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

    @classmethod
    def from_reference(
        cls,
        other: Self,
        amp_profiles: float,
        nb_importance_relation: int,
        nb_lex_order: int,
        rng: Generator,
    ):
        return cls(
            profiles=PerturbProfile(amp_profiles)(other.profiles, rng),
            importance_relation=PerturbImportanceRelation(nb_importance_relation)(
                other.importance_relation, rng
            ),
            lexicographic_order=PerturbLexOrder(
                len(other.profiles.alternatives), nb_lex_order
            )(other.lexicographic_order, rng),
        )


@dataclass
class RMPGroupModelImportanceProfilesLexicographic(
    GroupModel[RMPModel],
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
class RMPGroupModelImportanceProfiles(
    GroupModel[RMPModel],
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
    GroupModel[RMPModel],
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
    GroupModel[RMPModel],
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
    shared_params: RMPParamFlag,
) -> type[GroupModel[RMPModel]]:
    if RMPParamFlag.PROFILES in shared_params:
        if RMPParamFlag.IMPORTANCE_RELATION in shared_params:
            if RMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return RMPGroupModelImportanceProfilesLexicographic
            else:
                return RMPGroupModelImportanceProfiles
        else:
            if RMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return RMPGroupModelProfilesLexicographic
            else:
                return RMPGroupModelProfiles
    else:
        if RMPParamFlag.IMPORTANCE_RELATION in shared_params:
            if RMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return RMPGroupModelImportanceLexicographic
            else:
                return RMPGroupModelImportance
        else:
            if RMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return RMPGroupModelLexicographic
            else:
                return RMPGroupModel


def rmp_model(group_size: int, shared_params: RMPParamFlag = RMPParamFlag(0)):
    return RMPModel if group_size == 1 else rmp_group_model(shared_params)


def rmp_model_from_name(name: str) -> type[Model]:
    return eval(name)
