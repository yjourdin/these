import ast
from dataclasses import dataclass
from typing import Any

from ..field import (
    RandomField,
    random_field,
    random_group_field,
)
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..random import RNGParam, rng_
from ..utils import tolist
from .importance_relation import ImportanceRelation
from .profile import random_profiles


@random_field("profiles")
@dataclass
class ProfilesField(RandomField):
    profiles: NormalPerformanceTable

    @staticmethod
    def field_decode(o: Any):
        return NormalPerformanceTable(o)

    @staticmethod
    def field_encode(o: Any):  # type: ignore
        return tolist(o.data.values)

    @staticmethod
    def field_random(
        nb_profiles: int,
        nb_crit: int,
        rng: RNGParam = None,
        profiles_values: NormalPerformanceTable | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        return random_profiles(
            nb_profiles,
            nb_crit,
            rng,
            profiles_values,
        )


@random_field("profiles")
@dataclass(frozen=True)
class FrozenProfilesField(RandomField):
    profiles: tuple[tuple[float, ...], ...]

    @staticmethod
    def field_decode(o: Any):
        return tuple(tuple(profile) for profile in o)


@random_group_field(fieldname="profiles", fieldclass=ProfilesField)
@dataclass
class GroupProfilesField(RandomField):
    profiles: list[NormalPerformanceTable]


@random_field("importance_relation")
@dataclass
class ImportanceRelationField(RandomField):
    importance_relation: ImportanceRelation

    @staticmethod
    def field_decode(o: Any):
        return ImportanceRelation(
            [int(x) for x in o.values()],
            [frozenset(ast.literal_eval(label)) for label in o.keys()],
        )

    @staticmethod
    def field_encode(o: Any):
        return {str(list(label)): int(score) for label, score in o.items()}

    @staticmethod
    def field_random(nb_crit: int, rng: RNGParam = None, *args: Any, **kwargs: Any):
        return ImportanceRelation.random(nb_crit, rng)


@random_group_field(fieldname="importance_relation", fieldclass=ImportanceRelationField)
@dataclass
class GroupImportanceRelationField(RandomField):
    importance_relation: list[ImportanceRelation]


@random_field("lexicographic_order")
@dataclass
class LexicographicOrderField(RandomField):
    lexicographic_order: list[int]

    @staticmethod
    def field_random(nb_profiles: int, rng: RNGParam = None, *args: Any, **kwargs: Any):
        return tolist(rng_(rng).permutation(nb_profiles))


@random_field("profiles")
@dataclass(frozen=True)
class FrozenLexicographicOrderField(RandomField):
    lexicographic_order: tuple[int, ...]

    @staticmethod
    def field_decode(o: Any):
        return tuple(o)


@random_group_field(fieldname="lexicographic_order", fieldclass=LexicographicOrderField)
@dataclass
class GroupLexicographicOrderField(RandomField):
    lexicographic_order: list[list[int]]
