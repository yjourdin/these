from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from ..field import GeneratedField, generated_field, group_generated_field
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .capacity import Capacity, balanced_capacity, random_capacity
from .importance_relation import ImportanceRelation
from .profile import balanced_profiles, random_profiles


@generated_field("profiles")
@dataclass
class ProfilesField(GeneratedField):
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

    @staticmethod
    def field_balanced(
        nb_profiles: int,
        nb_crit: int,
        profiles_values: NormalPerformanceTable | None = None,
        *args,
        **kwargs,
    ):
        return balanced_profiles(
            nb_profiles,
            nb_crit,
            profiles_values,
        )


@group_generated_field(fieldname="profiles", fieldclass=ProfilesField)
@dataclass
class GroupProfilesField(GeneratedField):
    profiles: list[NormalPerformanceTable]


@generated_field("importance_relation")
@dataclass
class ImportanceRelationField(GeneratedField):
    importance_relation: ImportanceRelation

    @staticmethod
    def field_decode(o):
        return ImportanceRelation(
            np.array(o["data"]),
            [frozenset(label) for label in o["labels"]],
        )

    @staticmethod
    def field_encode(o):
        return {
            "labels": o.labels,
            "data": o.data.tolist(),
        }

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
        return ImportanceRelation.from_capacity(random_capacity(nb_crit, rng))

    @staticmethod
    def field_balanced(nb_crit: int, *args, **kwargs):
        return ImportanceRelation.from_capacity(balanced_capacity(nb_crit))


@group_generated_field(
    fieldname="importance_relation", fieldclass=ImportanceRelationField
)
@dataclass
class GroupImportanceRelationField(GeneratedField):
    importance_relation: list[ImportanceRelation]


@generated_field("capacity")
@dataclass
class CapacityField(GeneratedField):
    capacity: Capacity

    @staticmethod
    def field_decode(o):
        return {frozenset(k): v for k, v in o.items()}

    @staticmethod
    def field_encode(o):
        return {list(k): v for k, v in o.items()}

    @staticmethod
    def field_random(nb_crit: int, rng: Generator, *args, **kwargs):
        return random_capacity(nb_crit, rng)

    @staticmethod
    def field_balanced(nb_crit: int, *args, **kwargs):
        return balanced_capacity(nb_crit)


@group_generated_field(fieldname="capacity", fieldclass=CapacityField)
@dataclass
class GroupCapacityField(GeneratedField):
    capacity: list[Capacity]


@generated_field("lexicographic_order")
@dataclass
class LexicographicOrderField(GeneratedField):
    lexicographic_order: list[int]

    @staticmethod
    def field_random(nb_profiles: int, rng: Generator, *args, **kwargs):
        return rng.permutation(nb_profiles).tolist()

    @staticmethod
    def field_balanced(nb_profiles: int, rng: Generator, *args, **kwargs):
        return rng.permutation(nb_profiles).tolist()


@group_generated_field(
    fieldname="lexicographic_order", fieldclass=LexicographicOrderField
)
@dataclass
class GroupLexicographicOrderField(GeneratedField):
    lexicographic_order: list[list[int]]
