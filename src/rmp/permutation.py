from typing import Any


def swap(permutation: list[Any], i: int, j: int):
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


def adjacent_swap(permutation: list[Any], i: int):
    return swap(permutation, i, i + 1)


def all_max_adjacent_distance(permutation: list[Any], distance: int):
    k = len(permutation)
    adjacent_swap_indexes = [adjacent_swap(list(range(k)), i) for i in range(k - 1)]

    return all_max_distance({tuple(permutation)}, distance, adjacent_swap_indexes)


def all_max_distance(
    permutations: set[tuple[Any]], distance: int, op_indexes: list[list[int]]
):
    if distance == 0:
        return permutations

    new_permutations: set[tuple[Any]] = set()

    for indexes in op_indexes:
        for permutation in permutations:
            new_permutations.add(tuple(permutation[j] for j in indexes))
    return permutations | new_permutations
