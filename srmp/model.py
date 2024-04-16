from dataclasses import dataclass
from json import JSONDecoder, JSONEncoder, dumps, loads

from mcda.internal.core.scales import NormalScale

from abstract_model import Model
from performance_table.normal_performance_table import NormalPerformanceTable
from utils import print_list

from .normal_srmp import NormalSRMP


class SRMPEncoder(JSONEncoder):
    def default(self, o):
        try:
            return {
                "profiles": o.profiles.data.values.tolist(),
                "weights": list(o.weights),
                "lexicographic_order": list(o.lexicographic_order),
            }
        except TypeError:
            return super().default(o)


class SRMPDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if all(k in dct for k in ("profiles", "weights", "lexicographic_order")):
            profiles = NormalPerformanceTable(dct["profiles"])
            weights = list(dct["weights"])
            lexicographic_order = list(dct["lexicographic_order"])
            return SRMPModel(
                profiles=profiles,
                weights=weights,
                lexicographic_order=lexicographic_order,
            )
        return dct


@dataclass
class SRMPModel(Model[NormalScale]):
    profiles: NormalPerformanceTable
    weights: list[float]
    lexicographic_order: list[int]

    def __str__(self) -> str:
        return (
            f"{print_list(self.weights)}   "
            f"{print_list(self.profiles.data.to_numpy()[0])}   "
            f"{self.lexicographic_order.__str__()}"
        )

    @classmethod
    def from_json(cls, s):
        return loads(s, cls=SRMPDecoder)

    def to_json(self):
        return dumps(self, cls=SRMPEncoder, indent=4)

    def rank(self, performance_table):
        return NormalSRMP(
            performance_table,
            dict(zip(range(len(self.weights)), self.weights)),
            self.profiles,
            self.lexicographic_order,
        ).rank()
