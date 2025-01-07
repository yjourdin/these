import csv

from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure, R


def from_csv(csvfile) -> PreferenceStructure:
    reader = csv.reader(csvfile, "unix")
    relations: list[Relation] = []
    for line in reader:
        a, t, b = line
        a_i, b_i = int(a), int(b)
        match t:
            case "P":
                relations.append(P(a_i, b_i))
            case "I":
                relations.append(I(a_i, b_i))
            case "R":
                relations.append(R(a_i, b_i))
    return PreferenceStructure(relations, validate=False)


def to_csv(comparisons: PreferenceStructure, csvfile):
    writer = csv.writer(csvfile, "unix")
    writer.writerows([(r.a, r._RELATION_TYPE, r.b) for r in comparisons.relations])
