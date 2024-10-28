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
    Wm = W(m)
    k = 0
    S: list[float] = [0]

    if Wm >= sys.float_info.max:
        while Wm - S[-1] > 0 and (S[-1] - (S[-2] if len(S) > 1 else -1) > 0):
            k += 1
            S.append(S[-1] + (((k**m) // (2 ** (k + 1))) + 1))
    else:
        while Wm - S[-1] > delta and (S[-1] - (S[-2] if len(S) > 1 else -1) > 0):
            k += 1
            S.append(S[-1] + ((k**m) / (2 ** (k + 1))))

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


def random_ranking_from_blocks(m: int, k: int | None, rng: Generator):
    k = k or m

    return rng.integers(1, k, m, int, True)


def random_ranking_from_partial_sum(m: int, S: list[float], rng: Generator):
    K = random_nb_blocks(S, rng)

    return random_ranking_from_blocks(m, K, rng)


def random_ranking(m: int, rng: Generator, delta: float = 0.01):
    S = generate_partial_sum(m, delta)

    return random_ranking_from_partial_sum(m, S, rng)
