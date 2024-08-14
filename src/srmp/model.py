from dataclasses import dataclass
from enum import Enum

from mcda.internal.core.scales import NormalScale

from ..dataclass import GeneratedDataclass
from ..model import GroupModel, Model
from ..rmp.field import (
    GroupLexicographicOrderField,
    GroupProfilesField,
    LexicographicOrderField,
    ProfilesField,
)
from ..rmp.model import RMPParamEnum
from ..utils import print_list
from .field import GroupWeightsField, WeightsField
from .normal_srmp import NormalSRMP


class SRMPParamEnum(Enum):
    PROFILES = RMPParamEnum.PROFILES
    WEIGHTS = "weights"
    LEXICOGRAPHIC_ORDER = RMPParamEnum.LEXICOGRAPHIC_ORDER


@dataclass
class SRMPModel(
    Model[NormalScale],
    GeneratedDataclass,
    ProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __str__(self) -> str:
        return "\t".join(
            [
                print_list(self.weights),
                print_list(self.profiles.data.to_numpy()[0]),
                self.lexicographic_order.__str__(),
            ]
        )

    def rank(self, performance_table):
        return NormalSRMP(
            performance_table,
            dict(zip(range(len(self.weights)), self.weights)),
            self.profiles,
            self.lexicographic_order,
        ).rank()


@dataclass
class SRMPGroupModelWeightsProfilesLexicographic(
    GroupModel[NormalScale],
    GeneratedDataclass,
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


@dataclass
class SRMPGroupModelWeightsProfiles(
    GroupModel[NormalScale],
    GeneratedDataclass,
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
class SRMPGroupModelWeightsLexicographic(
    GroupModel[NormalScale],
    GeneratedDataclass,
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
class SRMPGroupModelProfilesLexicographic(
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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
    GroupModel[NormalScale],
    GeneratedDataclass,
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


def SRMP_model(shared_params: set[SRMPParamEnum]):
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
