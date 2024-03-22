from dataclasses import dataclass
from json import JSONDecoder, JSONEncoder, dumps, loads
from typing import Any

from mcda.core.scales import NormalScale

from abstract_model import Model
from performance_table.core import NormalPerformanceTable
from utils import print_list

from .core import NormalRMP


class RMPEncoder(JSONEncoder):
    def default(self, o):
        try:
            return {
                "profiles": o.profiles.data.values.tolist(),
                "capacities": {str(set(k)): v for k, v in o.capacities.items()},
                "lexicographic_order": list(o.lexicographic_order),
            }
        except TypeError:
            return super().default(o)


class RMPDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if all(k in dct for k in ("profiles", "capacities", "lexicographic_order")):
            profiles = NormalPerformanceTable(dct["profiles"])
            capacities = {frozenset(eval(k)): v for k, v in dct["capacities"].items()}
            lexicographic_order = list(dct["lexicographic_order"])
            return RMPModel(
                profiles=profiles,
                capacities=capacities,
                lexicographic_order=lexicographic_order,
            )
        return dct


@dataclass
class RMPModel(Model[NormalScale]):
    profiles: NormalPerformanceTable
    capacities: dict[frozenset[Any], float]
    lexicographic_order: list[int]

    def __str__(self) -> str:
        return (
            # f"{self.criteria_capacities.__str__()}   "
            f"{print_list(self.profiles.data.to_numpy()[0])}   "
            f"{self.lexicographic_order.__str__()}"
        )

    @classmethod
    def from_json(cls, s):
        return loads(s, cls=RMPDecoder)

    def to_json(self):
        return dumps(self, cls=RMPEncoder, indent=4)

    def rank(self, performance_table):
        return NormalRMP(
            performance_table,
            self.capacities,
            self.profiles,
            self.lexicographic_order,
        ).rank()
