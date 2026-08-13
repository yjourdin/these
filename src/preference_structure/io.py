import csv
from collections.abc import Iterable
from typing import Any

from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure, R


def from_csv(csvfile: Iterable[str]):
    reader = csv.reader(csvfile, "unix")
    relations: list[Relation] = []
    for line in reader:
        match line:
            case a, "P", b:
                relations.append(P(int(a), int(b)))
            case a, "I", b:
                relations.append(I(int(a), int(b)))
            case a, "R", b:
                relations.append(R(int(a), int(b)))
            case l:
                raise ValueError(f"Unknown line: {l}")
    return PreferenceStructure(relations, validate=False)


def to_csv(comparisons: PreferenceStructure, csvfile: Any):
    writer = csv.writer(csvfile, "unix")
    writer.writerows([str(r).split(" ") for r in comparisons])
