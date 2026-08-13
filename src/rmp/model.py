from dataclasses import dataclass
from enum import auto
from typing import Self, SupportsIndex

from src.model import FrozenModel, GroupModel, Model, ParamFlag
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.performance_table.type import PerformanceTableType
from src.random import RNGParam
from src.utils import print_list, tolist

from .field import (
    FrozenImportanceRelationField,
    FrozenLexicographicOrderField,
    FrozenProfilesField,
    GroupImportanceRelationField,
    GroupLexicographicOrderField,
    GroupProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
    ProfilesField,
)
from .importance_relation import ImportanceRelation
from .perturbations import PerturbImportanceRelation, PerturbLexOrder, PerturbProfile
from .rmp import NormalRMP


class RMPParamFlag(ParamFlag):
    NONE = 0
    PROFILES = auto()
    IMPORTANCE_RELATION = auto()
    LEXICOGRAPHIC_ORDER = auto()


@dataclass(unsafe_hash=True, slots=True)
class RMPModel(
    Model,
    ProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
):
    def __str__(self):
        return (
            print_list(self.profiles.data.to_numpy()[0])
            + "\t"
            + str(self.lexicographic_order.__str__())
        )

    def rank_numpy(self, performance_table: PerformanceTableType):
        if isinstance(performance_table, NormalPerformanceTable):
            return NormalRMP(
                performance_table,
                self.importance_relation,
                self.profiles,
                self.lexicographic_order,
            ).rank_numpy()
        else:
            raise TypeError("Performance table not normalized")

    @classmethod
    def from_reference(
        cls,
        other: Self,
        amp_profiles: float,
        amp_importance_relation: float,
        nb_lex_order: int,
        rng: RNGParam = None,
    ):
        return cls(
            profiles=PerturbProfile(amp_profiles)(other.profiles, rng),
            importance_relation=PerturbImportanceRelation(amp_importance_relation)(
                other.importance_relation, rng
            ),
            lexicographic_order=PerturbLexOrder(
                len(other.profiles.alternatives), nb_lex_order
            )(other.lexicographic_order, rng),
        )

    @property
    def frozen(self):
        return FrozenRMPModel(
            profiles=tuple(tuple(x) for x in tolist(self.profiles.data.to_numpy())),  # pyright: ignore[reportUnknownArgumentType]
            importance_relation=tuple(self.importance_relation.items()),
            lexicographic_order=tuple(self.lexicographic_order),
        )


@dataclass(frozen=True)
class FrozenRMPModel(
    FrozenModel[RMPModel],
    FrozenProfilesField,
    FrozenImportanceRelationField,
    FrozenLexicographicOrderField,
):
    @property
    def model(self):
        return RMPModel(
            profiles=NormalPerformanceTable(self.profiles),
            importance_relation=ImportanceRelation(
                *zip(*((v, k) for k, v in self.importance_relation))  # pyright: ignore[reportArgumentType]
            ),
            lexicographic_order=list(self.lexicographic_order),
        )


@dataclass
class RMPGroupModelImportanceProfilesLexicographic(
    GroupModel[RMPModel],
    ProfilesField,
    ImportanceRelationField,
    LexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return RMPModel(
            profiles=self.profiles,
            importance_relation=self.importance_relation,
            lexicographic_order=self.lexicographic_order,
        )

    @property
    def collective_model(self):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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
    def __getitem__(self, i: SupportsIndex | slice):
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


def rmp_model(group_size: int, shared_params: RMPParamFlag = RMPParamFlag.NONE):
    return RMPModel if group_size == 1 else rmp_group_model(shared_params)


def rmp_model_from_name(name: str) -> type[Model]:
    return eval(name)
