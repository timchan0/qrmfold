from collections.abc import Collection, Iterable
from itertools import chain, combinations
from typing import Literal

import numpy as np
from numpy import typing as npt


def complement(m: int, subset: Collection[int], start: int = 1):
    """Return the complement of a subset.

    :param m: The size of the universal set.
    :param subset: Subset of the universe.
    :param start: The starting index of the universal set (default is 1).
    :returns: The complement of ``subset`` within ``{start, ..., start + m - 1}``.
    """
    return set(range(start, start + m)).difference(subset)

# adapted from itertools recipes
def powerset(pairs: Collection[tuple[int, int]], max_cardinality: None | int = None):
    """Iterate subsequences of ``pairs`` from shortest to longest.

    :param pairs: Collection of pairs to take subsets from.
    :param max_cardinality: Optional cap on subset size.
    :returns: An iterator over tuples representing subsets.
    """
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    if max_cardinality is None:
        max_cardinality = len(pairs)
    return chain.from_iterable(combinations(pairs, r) for r in range(max_cardinality+1))


def extract_arguments(index: Literal[0, 1], l_subset: Iterable[tuple[int, int]]):
    """Extract the set of arguments at a given position from a pair iterable.

    :param index: Which element of each pair to extract (0 or 1).
    :param l_subset: Iterable of integer pairs.
    :returns: A set of extracted integers.
    """
    return {pair[index] for pair in l_subset}


def all_bitstrings(length: int) -> tuple[str, ...]:
    """Generate all bitstrings of a given length.

    :param length: Bitstring length.
    :returns: A tuple of all bitstrings of the given length.
    """
    if length == 0:
        return ('',)
    return tuple(np.binary_repr(k, width=length) for k in range(2**length))


def rref_gf2(matrix: list[npt.NDArray[np.bool_]]):
    """Compute the reduced row echelon form of a binary matrix over GF(2).

    :param matrix: List of row vectors.
    :returns: The reduced row echelon form as a numpy array of integers.
    """
    # TODO: use galois package instead https://stackoverflow.com/questions/56856378/fast-computation-of-matrix-rank-over-gf2
    array = np.array(matrix, dtype=np.int_)
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
"""Map from a Pauli string sign to the exponent of ``i``."""