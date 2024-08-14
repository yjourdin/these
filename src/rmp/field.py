from dataclasses import dataclass

import numpy as np

from ..field import GeneratedField, group_generated_field
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .capacity import Capacity, balanced_capacity, random_capacity
from .importance_relation import ImportanceRelation
from .profile import balanced_profiles, random_profiles


@dataclass
class ProfilesField(GeneratedField):
    profiles: NormalPerformanceTable

    @classmethod
    def json_to_dict(cls, dct: dict):
        super().json_to_dict(dct)
        if "profiles" in dct:
            dct["profiles"] = NormalPerformanceTable(dct["profiles"])
            return dct["profiles"]

    @classmethod
    def dict_to_json(cls, dct):
        super().dict_to_json(dct)
        if "profiles" in dct:
            dct["profiles"] = dct["profiles"].data.values.tolist()
            return dct["profiles"]

    @classmethod
    def random(
        cls,
        *args,
        **kwargs,
    ):
        super().random(*args, **kwargs)
        kwargs["profiles"] = random_profiles(
            kwargs["nb_profiles"],
            kwargs["nb_crit"],
            kwargs["rng"],
            kwargs["profiles_values"],
        )
        return kwargs["profiles"]

    @classmethod
    def balanced(
        cls,
        *args,
        **kwargs,
    ):
        super().balanced(*args, **kwargs)
        kwargs["profiles"] = balanced_profiles(
            kwargs["nb_profiles"],
            kwargs["nb_crit"],
            kwargs["profiles_values"],
        )
        return kwargs["profiles"]


# @dataclass
# class GroupProfilesField(
#     GeneratedField,
#     metaclass=GroupGeneratedField,
#     fieldname="profiles",
#     fieldclass=ProfilesField,
# ):
#     profiles: list[NormalPerformanceTable]


@group_generated_field(fieldname="profiles", fieldclass=ProfilesField)
@dataclass
class GroupProfilesField(GeneratedField):
    profiles: list[NormalPerformanceTable]


@dataclass
class ImportanceRelationField(GeneratedField):
    importance_relation: ImportanceRelation

    @classmethod
    def json_to_dict(cls, dct: dict):
        super().json_to_dict(dct)
        if "importance_relation" in dct:
            dct["importance_relation"] = ImportanceRelation(
                np.array(dct["importance_relation"]["data"]),
                [frozenset(label) for label in dct["importance_relation"]["labels"]],
            )
            return dct["importance_relation"]

    @classmethod
    def dict_to_json(cls, dct):
        super().dict_to_json(dct)
        if "importance_relation" in dct:
            dct["importance_relation"] = (
                {
                    "labels": dct["importance_relation"].labels,
                    "data": dct["importance_relation"].data.tolist(),
                },
            )
            return dct["importance_relation"]

    @classmethod
    def random(cls, *args, **kwargs):
        super().random(*args, **kwargs)
        kwargs["importance_relation"] = ImportanceRelation.from_capacity(
            random_capacity(kwargs["nb_crit"], kwargs["rng"])
        )
        return kwargs["importance_relation"]

    @classmethod
    def balanced(cls, *args, **kwargs):
        super().balanced(*args, **kwargs)
        kwargs["importance_relation"] = ImportanceRelation.from_capacity(
            balanced_capacity(kwargs["nb_crit"])
        )
        return kwargs["importance_relation"]


# @dataclass
# class GroupImportanceRelationField(
#     GeneratedField,
#     metaclass=GroupGeneratedField,
#     fieldname="importance_relation",
#     fieldclass=ImportanceRelationField,
# ):
#     importance_relation: list[ImportanceRelation]


@group_generated_field(
    fieldname="importance_relation", fieldclass=ImportanceRelationField
)
@dataclass
class GroupImportanceRelationField(GeneratedField):
    importance_relation: list[ImportanceRelation]


@dataclass
class CapacityField(GeneratedField):
    capacity: Capacity

    @classmethod
    def json_to_dict(cls, dct: dict):
        super().json_to_dict(dct)
        if "capacity" in dct:
            dct["capacity"] = {frozenset(k): v for k, v in dct["capacity"].items()}
            return dct["capacity"]

    @classmethod
    def dict_to_json(cls, dct):
        super().dict_to_json(dct)
        if "capacity" in dct:
            dct["capacity"] = {list(k): v for k, v in dct["capacity"].items()}
            return dct["capacity"]

    @classmethod
    def random(cls, *args, **kwargs):
        super().random(*args, **kwargs)
        kwargs["capacity"] = random_capacity(kwargs["nb_crit"], kwargs["rng"])
        return kwargs["capacity"]

    @classmethod
    def balanced(cls, *args, **kwargs):
        super().balanced(*args, **kwargs)
        kwargs["capacity"] = balanced_capacity(kwargs["nb_crit"])
        return kwargs["capacity"]


# @dataclass
# class GroupCapacityField(
#     GeneratedField,
#     metaclass=GroupGeneratedField,
#     fieldname="capacity",
#     fieldclass=CapacityField,
# ):
#     capacity: list[Capacity]


@group_generated_field(fieldname="capacity", fieldclass=CapacityField)
@dataclass
class GroupCapacityField(GeneratedField):
    capacity: list[Capacity]


@dataclass
class LexicographicOrderField(GeneratedField):
    lexicographic_order: list[int]

    @classmethod
    def random(cls, *args, **kwargs):
        super().random(*args, **kwargs)
        kwargs["lexicographic_order"] = (
            kwargs["rng"].permutation(kwargs["nb_profiles"]).tolist()
        )
        return kwargs["lexicographic_order"]

    @classmethod
    def balanced(cls, *args, **kwargs):
        super().balanced(*args, **kwargs)
        kwargs["lexicographic_order"] = (
            kwargs["rng"].permutation(kwargs["nb_profiles"]).tolist()
        )
        return kwargs["lexicographic_order"]


# @dataclass
# class GroupLexicographicOrderField(
#     GeneratedField,
#     metaclass=GroupGeneratedField,
#     fieldname="lexicographic_order",
#     fieldclass=LexicographicOrderField,
# ):
#     lexicographic_order: list[list[int]]


@group_generated_field(
    fieldname="lexicographic_order", fieldclass=LexicographicOrderField
)
@dataclass
class GroupLexicographicOrderField(GeneratedField):
    lexicographic_order: list[list[int]]
