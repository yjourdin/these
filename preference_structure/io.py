from mcda.relations import I, P, PreferenceStructure, R


def to_csv(comparisons: PreferenceStructure) -> str:
    s = ""
    for r in comparisons.relations:
        s += str(r).replace(" ", ",") + "\n"
    return s


def from_csv(s: str) -> PreferenceStructure:
    comparisons = PreferenceStructure()
    for line in s.splitlines():
        a, t, b = line.split(",")
        a_i, b_i = int(a), int(b)
        match t:
            case "P":
                comparisons._relations.append(P(a_i, b_i))
            case "I":
                comparisons._relations.append(I(a_i, b_i))
            case "R":
                comparisons._relations.append(R(a_i, b_i))
    return comparisons
