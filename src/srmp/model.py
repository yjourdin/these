from dataclasses import dataclass, replace
from enum import auto
from typing import Self, SupportsIndex, cast

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..enum import ParamFlag
from ..model import FrozenModel, GroupModel, Model
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..performance_table.type import PerformanceTableType
from ..rmp.field import (
    FrozenLexicographicOrderField,
    FrozenProfilesField,
    GroupLexicographicOrderField,
    GroupProfilesField,
    LexicographicOrderField,
    ProfilesField,
)
from ..rmp.perturbations import PerturbLexOrder, PerturbProfile
from ..srmp.perturbations import PerturbWeight
from ..utils import print_list, tolist
from .field import FrozenWeightsField, GroupWeightsField, WeightsField
from .normal_srmp import NormalSRMP


class SRMPParamFlag(ParamFlag):
    PROFILES = auto()
    WEIGHTS = auto()
    LEXICOGRAPHIC_ORDER = auto()


@dataclass
class SRMPModel(
    Model,
    ProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __str__(self) -> str:
        return "\t".join([
            print_list(list(self.weights)),
            print_list(self.profiles.data.to_numpy()[0]),
            print_list(self.lexicographic_order),
        ])

    def rank_numpy(self, performance_table: PerformanceTableType):
        return NormalSRMP(
            performance_table,
            self.weights,
            self.profiles,
            self.lexicographic_order,
        ).rank_numpy()

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
            lexicographic_order=PerturbLexOrder(
                len(other.profiles.alternatives), nb_lex_order
            )(other.lexicographic_order, rng),
        )

    @property
    def frozen(self):
        return FrozenSRMPModel(
            profiles=tuple(
                tuple(cast(list[float], x))
                for x in tolist(self.profiles.data.to_numpy())
            ),
            # weights=tuple(tolist(self.weights)),
            weights=self.weights,
            lexicographic_order=tuple(self.lexicographic_order),
        )


@dataclass(frozen=True)
class FrozenSRMPModel(
    FrozenModel[SRMPModel],
    FrozenProfilesField,
    FrozenWeightsField,
    FrozenLexicographicOrderField,
):
    @property
    def model(self):
        return SRMPModel(
            profiles=NormalPerformanceTable(self.profiles),
            weights=np.array(self.weights),
            lexicographic_order=list(self.lexicographic_order),
        )

    def replace_weights(self, weights: npt.NDArray[np.float64]):
        new = replace(self, weights=weights)
        return new


@dataclass
class SRMPGroupModelWeightsProfilesLexicographic(
    GroupModel[SRMPModel],
    ProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
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
class SRMPGroupModelWeightsProfiles(
    GroupModel[SRMPModel],
    ProfilesField,
    WeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelWeightsLexicographic(
    GroupModel[SRMPModel],
    GroupProfilesField,
    WeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights,
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModelProfilesLexicographic(
    GroupModel[SRMPModel],
    ProfilesField,
    GroupWeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModelWeights(
    GroupModel[SRMPModel],
    GroupProfilesField,
    WeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights,
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelProfiles(
    GroupModel[SRMPModel],
    ProfilesField,
    GroupWeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles,
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order[i],
        )


@dataclass
class SRMPGroupModelLexicographic(
    GroupModel[SRMPModel],
    GroupProfilesField,
    GroupWeightsField,
    LexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order,
        )


@dataclass
class SRMPGroupModel(
    GroupModel[SRMPModel],
    GroupProfilesField,
    GroupWeightsField,
    GroupLexicographicOrderField,
):
    def __getitem__(self, i: SupportsIndex | slice):
        return SRMPModel(
            profiles=self.profiles[i],
            weights=self.weights[i],
            lexicographic_order=self.lexicographic_order[i],
        )


def srmp_group_model(
    shared_params: SRMPParamFlag,
):
    if SRMPParamFlag.PROFILES in shared_params:
        if SRMPParamFlag.WEIGHTS in shared_params:
            if SRMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return SRMPGroupModelWeightsProfilesLexicographic
            else:
                return SRMPGroupModelWeightsProfiles
        else:
            if SRMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return SRMPGroupModelProfilesLexicographic
            else:
                return SRMPGroupModelProfiles
    else:
        if SRMPParamFlag.WEIGHTS in shared_params:
            if SRMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return SRMPGroupModelWeightsLexicographic
            else:
                return SRMPGroupModelWeights
        else:
            if SRMPParamFlag.LEXICOGRAPHIC_ORDER in shared_params:
                return SRMPGroupModelLexicographic
            else:
                return SRMPGroupModel


def srmp_model(group_size: int, shared_params: SRMPParamFlag = SRMPParamFlag(0)):
    return SRMPModel if group_size == 1 else srmp_group_model(shared_params)


def srmp_model_from_name(name: str) -> type[Model]:
    return eval(name)
