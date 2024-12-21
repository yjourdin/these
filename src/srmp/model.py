from collections.abc import Container
from dataclasses import dataclass
from typing import Self

from numpy.random import Generator

from ..dataclass import RandomDataclass
from ..enum_base import StrEnum
from ..model import GroupModel, Model
from ..rmp.field import (
    GroupLexicographicOrderField,
    GroupProfilesField,
    LexicographicOrderField,
    ProfilesField,
)
from ..rmp.model import RMPParamEnum
from ..rmp.perturbations import PerturbLexOrder, PerturbProfile
from ..srmp.perturbations import PerturbWeight
from ..utils import print_list
from .field import GroupWeightsField, WeightsField
from .normal_srmp import NormalSRMP


class SRMPParamEnum(StrEnum):
    PROFILES = RMPParamEnum.PROFILES.value
    WEIGHTS = "weights"
    LEXICOGRAPHIC_ORDER = RMPParamEnum.LEXICOGRAPHIC_ORDER.value


@dataclass(unsafe_hash=True)
class SRMPModel(  # type: ignore
    Model,
    RandomDataclass,
    ProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __str__(self) -> str:
        return "\t".join(
            [
                print_list(self.weights.tolist()),
                print_list(self.profiles.data.to_numpy()[0]),
                self.lexicographic_order.__str__(),
            ]
        )

    def rank(self, performance_table):
        return NormalSRMP(
            performance_table,
            self.weights,
            self.profiles,
            self.lexicographic_order,
        ).rank()

    @classmethod
    def from_reference(
        cls,
        other: Self,
        amp_profiles: float,
        amp_weights: float,
        nb_lex_order: int,
        rng: Generator,
    ):
        return cls(
            profiles=PerturbProfile(amp_profiles)(other.profiles, rng),
            weights=PerturbWeight(amp_weights)(other.weights, rng),
            lexicographic_order=PerturbLexOrder(nb_lex_order)(
                other.lexicographic_order, rng
            ),
        )


@dataclass
class SRMPGroupModelWeightsProfilesLexicographic(  # type: ignore
    GroupModel[SRMPModel],
    RandomDataclass,
    ProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights,
            lexicographic_order=self.lexicographic_order,
        )

    @property
    def collective_model(self) -> Model:
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights,
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModelWeightsProfiles(  # type: ignore
    GroupModel[SRMPModel],
    RandomDataclass,
    ProfilesField,
    WeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelWeightsLexicographic(  # type: ignore
    GroupModel[SRMPModel],
    RandomDataclass,
    GroupProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights,
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModelProfilesLexicographic(  # type: ignore
    GroupModel[SRMPModel],
    RandomDataclass,
    ProfilesField,
    GroupWeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModelWeights(
    GroupModel[SRMPModel],
    RandomDataclass,
    GroupProfilesField,
    WeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelProfiles(
    GroupModel[SRMPModel],
    RandomDataclass,
    ProfilesField,
    GroupWeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelLexicographic(
    GroupModel[SRMPModel],
    RandomDataclass,
    GroupProfilesField,
    GroupWeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModel(
    GroupModel[SRMPModel],
    RandomDataclass,
    GroupProfilesField,
    GroupWeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order[i],
        )


def srmp_group_model(
    shared_params: Container[SRMPParamEnum],
) -> type[GroupModel[SRMPModel]]:
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


def srmp_model(group_size: int, shared_params: Container[SRMPParamEnum] = set()):
    if group_size == 1:
        return SRMPModel
    else:
        return srmp_group_model(shared_params)


def srmp_model_from_name(name: str) -> type[Model]:
    return eval(name)
