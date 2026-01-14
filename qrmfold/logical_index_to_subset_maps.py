import itertools
import math


def default(m: int, start_index: int = 0):
    """Return the lexicographic map from logical index to subset of [m] of cardinality m/2."""
    return dict(enumerate(itertools.combinations(range(1, m+1), m//2), start=start_index))


def alternative(m: int, start_index: int = 0):
    """Return the map from logical index to subset of [m] of cardinality m/2
    where the first half is lexicographically ordered
    and the second half is the complement of the first half.
    """
    map_: dict[int, tuple[int, ...]] = {}
    logical_qubit_count = math.comb(m, m//2)
    one_to_m = range(1, m+1)
    for index, subset in zip(
        range(start_index, start_index + logical_qubit_count//2),
        itertools.combinations(one_to_m, m//2),
    ):
        map_[index] = subset
        complement = tuple(k for k in one_to_m if k not in subset)
        map_[logical_qubit_count//2 + index] = complement
    return map_