from collections import deque
from collections.abc import Collection, Iterable
from itertools import chain, combinations, islice
from typing import Literal

import numpy as np


# adapted from itertools recipes
def powerset(pairs: Collection[tuple[int, int]], max_cardinality: None | int = None):
    """Subsequences of the collection `pairs` from shortest to longest."""
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    if max_cardinality is None:
        max_cardinality = len(pairs)
    return chain.from_iterable(combinations(pairs, r) for r in range(max_cardinality+1))


def extract_arguments(index: Literal[0, 1], l_subset: Iterable[tuple[int, int]]):
    return {pair[index] for pair in l_subset}


def sliding_window(iterable: Iterable[set[int]], n: int):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 3) → ABC BCD CDE DEF EFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def all_bitstrings(length: int) -> tuple[str, ...]:
    """Generate a tuple of all bitstrings of the given length."""
    if length == 0:
        return ('',)
    return tuple(np.binary_repr(k, width=length) for k in range(2**length))


def rref_gf2(matrix):
    """Compute the reduced row echelon form of a binary matrix over GF(2)."""
    # TODO: use galois package instead https://stackoverflow.com/questions/56856378/fast-computation-of-matrix-rank-over-gf2
    array = np.array(matrix, dtype=int)
    rows, cols = array.shape
    pivot_row = 0

    for j in range(cols):
        if pivot_row >= rows:
            break

        # Find a pivot (a '1' in the current column)
        i = pivot_row
        while i < rows and array[i, j] == 0:
            i += 1

        if i < rows:
            # Swap rows if the pivot is not in the current pivot_row
            if i != pivot_row:
                array[[i, pivot_row]] = array[[pivot_row, i]]

            # Eliminate other 1s in the current column using XOR (modulo 2 addition)
            for k in range(rows):
                if k != pivot_row and array[k, j] == 1:
                    array[k, :] = array[k, :] ^ array[pivot_row, :]

            pivot_row += 1

    return array


sign_to_power: dict[complex, int] = {1: 0, 1j: 1, -1: 2, -1j: 3}
"""Map from Pauli string sign to exponent of i."""