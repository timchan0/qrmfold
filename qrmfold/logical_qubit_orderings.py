import itertools
import math

from qrmfold.utils import complement


def lexicographic(bit_count: int, start_index: int = 0):
    """Return the lexicographic logical qubit ordering.

    :param bit_count: The number m of bits of the
        underlying classical Reed-Muller code; must be even.
    :param start_index: Starting logical index.
    :returns ordering: A map from logical qubit index to
        a unique subset of [m] of cardinality m/2.
    """
    return dict(enumerate((
        set(subset) for subset in
        itertools.combinations(range(1, bit_count+1), bit_count//2)
    ), start=start_index))


def canonical(bit_count: int, start_index: int = 0):
    """Return the canonical logical qubit ordering.

    The first half of the logical qubits indices is lexicographically ordered.
    The second half is the complement of the first half.

    :param bit_count: The number m of bits of the
        underlying classical Reed-Muller code; must be even.
    :param start_index: Starting logical index.
    :returns ordering: A map from logical qubit index to
        a unique subset of [m] of cardinality m/2.
    """
    map_: dict[int, set[int]] = {}
    logical_qubit_count = math.comb(bit_count, bit_count//2)
    one_to_m = range(1, bit_count+1)
    for index, subset in zip(
        range(start_index, start_index + logical_qubit_count//2),
        itertools.combinations(one_to_m, bit_count//2),
    ):
        map_[index] = set(subset)
        map_[logical_qubit_count//2 + index] = complement(bit_count, subset)
    return map_