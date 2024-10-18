from functools import cache

from numpy.random import Generator


@cache
def w(m: int, k: int):
    if k > m:
        return 0
    if k == 1:
        return 1
    return k * (w(m - 1, k) + w(m - 1, k - 1))


def W(m: int):
    return sum(w(m, k) for k in range(1, m + 1))


def generate_partial_sum(m: int, delta: float = 0.01):
    Wm = W(m)
    k = 0
    S: list[float] = [0]

    while Wm - S[-1] >= delta:
        k += 1
        S.append(S[-1] + (k**m) / (2 ** (k + 1)))
    S[-1] = Wm

    return S


def random_nb_blocks(S: list[float], rng: Generator):
    Wm = S[-1]
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
