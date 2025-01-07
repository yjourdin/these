import ast
from dataclasses import dataclass

from numpy.random import Generator

from ..field import (
    RandomField,
    random_field,
    random_group_field,
)
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .importance_relation import ImportanceRelation
from .profile import random_profiles


@random_field("profiles")
@dataclass
class ProfilesField(RandomField):
    profiles: NormalPerformanceTable

    @staticmethod
    def field_decode(o):
        return NormalPerformanceTable(o)

    @staticmethod
    def field_encode(o):
        return o.data.values.tolist()

    @staticmethod
    def field_random(
        nb_profiles: int,
        nb_crit: int,
        rng: Generator,
        profiles_values: NormalPerformanceTable | None = None,
        *args,
        **kwargs,
    ):
        return random_profiles(
            nb_profiles,
            nb_crit,
            rng,
            profiles_values,
        )


@random_group_field(fieldname="profiles", fieldclass=ProfilesField)
@dataclass
class GroupProfilesField(RandomField):
    profiles: list[NormalPerformanceTable]


@random_field("importance_relation")
@dataclass
class ImportanceRelationField(RandomField):
    importance_relation: ImportanceRelation

    @staticmethod
    def field_decode(o):
        return ImportanceRelation(
            o.values(),
            [frozenset(ast.literal_eval(label)) for label in o.keys()],
        )

    @staticmethod
    def field_encode(o):
        return {str(list(label)): score for label, score in o.items()}

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
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
    def field_random(nb_profiles: int, rng: Generator, *args, **kwargs):
        return rng.permutation(nb_profiles).tolist()


@random_group_field(fieldname="lexicographic_order", fieldclass=LexicographicOrderField)
@dataclass
class GroupLexicographicOrderField(RandomField):
    lexicographic_order: list[list[int]]
