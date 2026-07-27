from typing import Any


def swap(permutation: list[Any], i: int, j: int):
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


def adjacent_swap(permutation: list[Any], i: int):
    return swap(permutation, i, i + 1)


def all_max_adjacent_distance(permutation: list[Any], distance: int):
    k = len(permutation)
    adjacent_swap_indexes = [adjacent_swap(list(range(k)), i) for i in range(k - 1)]

    permutations = {tuple(permutation)}
    last_permutations = permutations

    for _ in range(distance):
        last_permutations = all_adjacent(last_permutations, adjacent_swap_indexes)
        permutations |= last_permutations

    return permutations


def all_adjacent(permutations: set[tuple[Any]], op_indexes: list[list[int]]):
    new_permutations: set[tuple[Any]] = set()

    for indexes in op_indexes:
        for permutation in permutations:
            new_permutations.add(tuple(permutation[j] for j in indexes))
    return permutations | new_permutations
