from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

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
        nb_profiles: int,
        nb_crit: int,
        rng: Generator,
        profiles_values: NormalPerformanceTable | None = None,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().random(
            nb_profiles=nb_profiles,
            nb_crit=nb_crit,
            rng=rng,
            profiles_values=profiles_values,
            init_dict=init_dict,
            *args,
            **kwargs,
        )
        init_dict["profiles"] = random_profiles(
            nb_profiles,
            nb_crit,
            rng,
            profiles_values,
        )
        return init_dict["profiles"]

    @classmethod
    def balanced(
        cls,
        nb_profiles: int,
        nb_crit: int,
        profiles_values: NormalPerformanceTable | None = None,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().balanced(
            nb_profiles=nb_profiles,
            nb_crit=nb_crit,
            profiles_values=profiles_values,
            init_dict=init_dict,
            *args,
            **kwargs,
        )
        init_dict["profiles"] = balanced_profiles(
            nb_profiles,
            nb_crit,
            profiles_values,
        )
        return init_dict["profiles"]


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
    def random(
        cls,
        nb_crit: int,
        rng: Generator,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().random(nb_crit=nb_crit, rng=rng, init_dict=init_dict, *args, **kwargs)
        init_dict["importance_relation"] = ImportanceRelation.from_capacity(
            random_capacity(nb_crit, rng)
        )
        return init_dict["importance_relation"]

    @classmethod
    def balanced(cls, nb_crit: int, init_dict: dict[str, Any] = {}, *args, **kwargs):
        super().balanced(nb_crit=nb_crit, init_dict=init_dict, *args, **kwargs)
        init_dict["importance_relation"] = ImportanceRelation.from_capacity(
            balanced_capacity(nb_crit)
        )
        return init_dict["importance_relation"]


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
    def random(
        cls,
        nb_crit: int,
        rng: Generator,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().random(nb_crit=nb_crit, rng=rng, init_dict=init_dict, *args, **kwargs)
        init_dict["capacity"] = random_capacity(nb_crit, rng)
        return init_dict["capacity"]

    @classmethod
    def balanced(cls, nb_crit: int, init_dict: dict[str, Any] = {}, *args, **kwargs):
        super().balanced(nb_crit=nb_crit, init_dict=init_dict, *args, **kwargs)
        init_dict["capacity"] = balanced_capacity(nb_crit)
        return init_dict["capacity"]


@group_generated_field(fieldname="capacity", fieldclass=CapacityField)
@dataclass
class GroupCapacityField(GeneratedField):
    capacity: list[Capacity]


@dataclass
class LexicographicOrderField(GeneratedField):
    lexicographic_order: list[int]

    @classmethod
    def random(
        cls,
        nb_profiles: int,
        rng: Generator,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().random(
            nb_profiles=nb_profiles, rng=rng, init_dict=init_dict, *args, **kwargs
        )
        init_dict["lexicographic_order"] = rng.permutation(nb_profiles).tolist()
        return init_dict["lexicographic_order"]

    @classmethod
    def balanced(
        cls,
        nb_profiles: int,
        rng: Generator,
        init_dict: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        super().balanced(
            nb_profiles=nb_profiles, rng=rng, init_dict=init_dict, *args, **kwargs
        )
        init_dict["lexicographic_order"] = rng.permutation(nb_profiles).tolist()
        return init_dict["lexicographic_order"]


@group_generated_field(
    fieldname="lexicographic_order", fieldclass=LexicographicOrderField
)
@dataclass
class GroupLexicographicOrderField(GeneratedField):
    lexicographic_order: list[list[int]]
