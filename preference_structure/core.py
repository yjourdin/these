from mcda.core.relations import (
    IncomparableRelation,
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
    Relation,
)


def to_csv(comparisons: PreferenceStructure) -> str:
    s = ""
    for r in comparisons.relations:
        s += str(r).replace(" ", ",") + "\n"
    return s


def from_csv(s: str) -> PreferenceStructure:
    comparisons = PreferenceStructure()
    relations: list[Relation] = []
    for line in s.splitlines():
        a, t, b = line.split(",")
        a_i, b_i = int(a), int(b)
        match t:
            case "P":
                relations.append(PreferenceRelation(a_i, b_i))
            case "I":
                relations.append(IndifferenceRelation(a_i, b_i))
            case "R":
                relations.append(IncomparableRelation(a_i, b_i))
    comparisons._relations = relations
    return comparisons
