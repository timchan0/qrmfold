import itertools
import math

from qrmfold._utils import complement


def lexicographic(num_variables: int, start_index: int = 0):
    """Return the lexicographic logical qubit ordering.

    :param num_variables: The number m of variables of the
        underlying classical Reed-Muller code; must be even.
    :param start_index: Starting logical index.
    :returns ordering: A map from logical qubit index to
        a unique subset of [m] of cardinality m/2.
    """
    return dict(enumerate((
        set(subset) for subset in
        itertools.combinations(range(1, num_variables+1), num_variables//2)
    ), start=start_index))


def canonical(num_variables: int, start_index: int = 0):
    """Return the canonical logical qubit ordering.

    The first half of the logical qubits indices is lexicographically ordered.
    The second half is the complement of the first half.

    :param num_variables: The number m of variables of the
        underlying classical Reed-Muller code; must be even.
    :param start_index: Starting logical index.
    :returns ordering: A map from logical qubit index to
        a unique subset of [m] of cardinality m/2.
    """
    map_: dict[int, set[int]] = {}
    logical_qubit_count = math.comb(num_variables, num_variables//2)
    one_to_m = range(1, num_variables+1)
    for index, subset in zip(
        range(start_index, start_index + logical_qubit_count//2),
        itertools.combinations(one_to_m, num_variables//2),
    ):
        map_[index] = set(subset)
        map_[logical_qubit_count//2 + index] = complement(num_variables, subset)
    return map_