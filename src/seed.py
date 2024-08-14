from numpy.random import Generator


def seed(rng: Generator) -> int:
    return rng.integers(2**63)


def seeds(rng: Generator, nb: int = 1) -> list[int]:
    return rng.integers(2**63, size=nb).tolist()
