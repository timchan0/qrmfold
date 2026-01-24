import itertools
import math

from qrmfold.utils import complement


def default(m: int, start_index: int = 0):
    """Return the lexicographic logical qubit ordering.

    :param m: Universe size (must be even for use with the code).
    :param start_index: Starting logical index.
    :returns: A dict mapping each logical index to a subset of ``[m]`` of
        cardinality ``m/2``.
    """
    return dict(enumerate((
        set(subset) for subset in
        itertools.combinations(range(1, m+1), m//2)
    ), start=start_index))


def alternative(m: int, start_index: int = 0):
    """Return the alternative logical qubit ordering.

    The first half of the logical qubits is lexicographically ordered.
    The second half is the complement of the first half.

    :param m: Universe size.
    :param start_index: Starting logical index.
    :returns: A dict mapping each logical index to a subset of ``[m]`` of
        cardinality ``m/2``.
    """
    map_: dict[int, set[int]] = {}
    logical_qubit_count = math.comb(m, m//2)
    one_to_m = range(1, m+1)
    for index, subset in zip(
        range(start_index, start_index + logical_qubit_count//2),
        itertools.combinations(one_to_m, m//2),
    ):
        map_[index] = set(subset)
        map_[logical_qubit_count//2 + index] = complement(m, subset)
    return map_