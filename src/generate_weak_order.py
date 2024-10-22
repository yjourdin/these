import sys
from functools import cache
from random import randint, seed

from numpy.random import Generator


@cache
def w(m: int, k: int):
    if k > m:
        return 0
    if k == 1:
        return 1
    return k * (w(m - 1, k) + w(m - 1, k - 1))


def W(m: int):
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(2 * m, recursion_limit))
    result = sum(w(m, k) for k in range(1, m + 1))
    sys.setrecursionlimit(recursion_limit)
    return result


def generate_partial_sum(m: int, delta: float = 0.01):
    # delta_decimal = Decimal(delta)
    # Wm = Decimal(W(m))
    Wm = W(m)
    k = 0
    S: list[float] = [0]

    if Wm >= sys.float_info.max:
        while Wm - S[-1] > 0 and (S[-1] - (S[-2] if len(S) > 1 else -1) > 0):
            k += 1
            S.append(S[-1] + (((k**m) // (2 ** (k + 1))) + 1))
            # print("it", len(S))
            # print("Wm", Wm)
            # print("S1", S[-1])
            # print("S2", S[-2])
            # print("km", k**m)
            # print("2k", 2 ** (k + 1))
            # print("frac", (k**m) // (2 ** (k + 1)) + 1)
            # print("diff", S[-1] - S[-2])
            # print("Wm diff", Wm - S[-1])
            # print("---")
    else:
        while Wm - S[-1] > delta and (S[-1] - (S[-2] if len(S) > 1 else -1) > 0):
            k += 1
            S.append(S[-1] + ((k**m) / (2 ** (k + 1))))
            # print("it", len(S))
            # print("Wm", Wm)
            # print("S1", S[-1])
            # print("S2", S[-2])
            # print("km", k**m)
            # print("2k", 2 ** (k + 1))
            # print("frac", (k**m) / (2 ** (k + 1)))
            # print("diff", S[-1] - S[-2])
            # print("Wm diff", Wm - S[-1])
            # print("---")

    S[-1] = Wm

    return S


def random_nb_blocks(S: list[float], rng: Generator):
    Wm = int(S[-1])
    if Wm >= sys.float_info.max:
        seed(rng.bit_generator.random_raw())
        Y = randint(0, Wm)
    else:
        Y = rng.uniform(0, Wm)

    for K, Sk in enumerate(S):
        if Sk >= Y:
            break

    return K


def random_ranking(alternatives: list, k: int | None, rng: Generator):
    m = len(alternatives)
    k = k or m

    return rng.integers(1, k, m, int, True)


def random_ranking_with_tie_from_partial_sum(
    alternatives: list, S: list[float], rng: Generator
):
    K = random_nb_blocks(S, rng)

    return random_ranking(alternatives, K, rng)


def random_ranking_with_tie(alternatives: list, rng: Generator, delta: float = 0.01):
    m = len(alternatives)
    S = generate_partial_sum(m, delta)

    return random_ranking_with_tie_from_partial_sum(alternatives, S, rng)
