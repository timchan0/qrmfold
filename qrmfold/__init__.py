from collections.abc import Iterable
import itertools
from functools import reduce
import math
from typing import Literal

from reedmuller.reedmuller import _vector_mult
import stim
import numpy as np


def logical_index_to_subset_default(m: int):
    """Return the lexicographic map from logical index to subset of [m] of cardinality m/2."""
    return dict(enumerate(itertools.combinations(range(1, m+1), m//2)))


def logical_index_to_subset_alternative(m: int):
    map_ = {}
    logical_qubit_count = math.comb(m, m//2)
    one_to_m = range(1, m+1)
    for index, subset in zip(
        range(logical_qubit_count//2),
        itertools.combinations(one_to_m, m//2),
    ):
        map_[index] = subset
        complement = tuple(k for k in one_to_m if k not in subset)
        map_[logical_qubit_count//2 + index] = complement
    return map_


class ReedMuller:
    """Inspired by `reedmuller.reedmuller.ReedMuller`."""

    def __init__(self, r: int, m: int):
        self.R = r
        self.M = m
        # Construct all of the x_i rows.
        self.basis = [self._construct_vector(i) for i in range(m)]
        # For every s-set S for all 0 <= s <= r, create the row that is the product of the x_j vectors for j in S.
        self.generator_matrix: list[list[int]] = [reduce(_vector_mult, [self.basis[i] for i in S], [1] * (2 ** m))
                              for s in range(r + 1)
                              for S in itertools.combinations(range(m), s)]
        
    def _construct_vector(self, i: int) -> list[int]:
        """Construct the vector for x_i of length 2^m, which has form:
        A string of 2^i 0s followed by 2^i 1s, repeated
        2^m / (2*2^i) = 2^{m-1}/2^i = 2^{m-1-i} times.
        NOTE: we must have 0 <= i < m."""
        return ([0] * 2**i + [1] * 2**i) * 2 ** (self.M-i-1)


class QuantumReedMuller:

    def __init__(
            self,
            m: int,
            logical_index_to_subset: None | dict[int, tuple[int, ...]] = None,
    ):
        """Input:
        * `m` the bit count of the binary vector labels (must be even).
        * `logical_index_to_subset` a 1-to-1 map from logical qubit index to subset of [m] of cardinality m/2.
        If not specified, the subsets are ordered lexicographically.
        """
        if m % 2:
            raise ValueError("m must be even")
        self.M = m
        r = m//2 - 1
        self.classical = ReedMuller(r, m)
        self.stabilizer_generators = {
            basis: [stim.PauliString(basis if p else 'I' for p in g) for g in self.classical.generator_matrix]
            for basis in ('X', 'Z')
        }
        self.logical_x_supports: dict[tuple[int, ...], list[int]] = {
            subset: _vector_mult(*(self.classical.basis[i-1] for i in subset))
            for subset in itertools.combinations(range(1, m+1), m//2)
        }
        self.logical_operators: dict[tuple[int, ...], list[stim.PauliString]] = {}
        for subset, support in self.logical_x_supports.items():
            x = stim.PauliString('X' if p else 'I' for p in support)
            complement = tuple(k for k in range(1, m+1) if k not in subset)
            z = stim.PauliString('Z' if p else 'I' for p in self.logical_x_supports[complement])
            self.logical_operators[subset] = [x, z]
        self.logical_index_to_subset = logical_index_to_subset_default(m) \
            if logical_index_to_subset is None else logical_index_to_subset

    def print(self):
        for basis in ('X', 'Z'):
            print(f'{basis} stabilizer generators:')
            for g in self.stabilizer_generators[basis]:
                print(g)
        print('logical operators:')
        digit_count = math.ceil(math.log10(len(self.logical_x_supports)))
        for logical_index, subset in self.logical_index_to_subset.items():
            print(f'{logical_index:{digit_count}d}', subset, [str(pauli) for pauli in self.logical_operators[subset]])

    def stabilizer_generators_bsf(self):
        """Return the stabilizer generators in binary symplectic form as a numpy array."""
        stabilizers_bsf: list[np.ndarray] = []
        for basis_generators in self.stabilizer_generators.values():
            for generator in basis_generators:
                xs, zs = generator.to_numpy()
                stabilizers_bsf.append(np.append(xs, zs))
        return np.array(stabilizers_bsf)
    
    def trivial_automorphism(self):
        """The trivial automorphism that maps every element to itself."""
        return Automorphism(self.M, [])

    def automorphism(self, type_: Literal['P', 'Q'], pairs: Iterable[tuple[int, int]]):
        """The automorphism P(i, j) or Q(i, j).

        P(i, j) swaps basis vectors i and j (1-indexed).
        Q(i, j) adds basis vector j onto basis vector i (1-indexed).

        Input:
        * `type_` 'P' or 'Q'.
        * `pairs` an iterable of pairs of integers (i, j) with 1 <= i, j <= M.
        Each integer in `pairs` must be distinct.
        
        Output:
        * A list whose kth element is the position of binary vector k after the automorphism.
        """
        automorphism = self.trivial_automorphism()
        exhausted_indices: set[int] = set()
        for i, j in pairs:
            if not ((1 <= i <= self.M) and (1 <= j <= self.M)):
                raise ValueError(f"i and j must be between 1 and {self.M}, inclusive")
            if (pair:={i, j}) & exhausted_indices:
                raise ValueError("Each integer in pairs must be distinct")
            exhausted_indices.update(pair)
            if type_ == 'P':
                automorphism.update(_p_automorphism(self.M, i, j))
            elif type_ == 'Q':
                automorphism.update(_q_automorphism(self.M, i, j))
            else:
                raise ValueError("type_ must be 'P' or 'Q'")
        return automorphism


def _p_automorphism(m: int, i: int, j: int):
    """Swap basis vectors i and j (1-indexed).
    
    Output:
    * A set of pairs of bit indices that are swapped.
    The first element in each pair has a 0 in position i and a 1 in position j,
    and the second element has a 1 in position i and a 0 in position j.
    """
    result: list[tuple[str, str]] = []
    for a, b, c in _get_residual_slices(m, i, j):
        s1 = a + '0' + b + '1' + c
        s2 = a + '1' + b + '0' + c
        result.append((s1, s2))
    return result


def _q_automorphism(m: int, i: int, j: int):
    """Add basis vector j onto basis vector i (1-indexed).
    
    Output:
    * A set of pairs of bit indices that are swapped.
    The first element in each pair has a 0 in position i and a 1 in position j,
    and the second element has a 1 in position i and a 1 in position j.
    """
    if i < j:
        s1_larger, s1_smaller = '1', '0'
    else:
        s1_larger, s1_smaller = '0', '1'
    result: list[tuple[str, str]] = []
    for a, b, c in _get_residual_slices(m, i, j):
        s1 = a + s1_larger + b + s1_smaller + c
        s2 = a + '1' + b + '1' + c
        result.append((s1, s2))
    return result


def _get_residual_slices(m: int, i: int, j: int):
    smaller, larger = min(i, j), max(i, j)
    residual_1 = _all_bitstrings(m-larger)
    residual_2 = _all_bitstrings(larger-1-smaller)
    residual_3 = _all_bitstrings(smaller-1)
    return itertools.product(residual_1, residual_2, residual_3)


def _all_bitstrings(length: int) -> tuple[str, ...]:
    """Generate a tuple of all bitstrings of the given length."""
    if length == 0:
        return ('',)
    return tuple(np.binary_repr(k, width=length) for k in range(2**length))
    

class Automorphism:
    """An automorphism of of 2^m elements represented as a list of positions.
    
    Instance attributes:
    * `M` the bit count of the binary vector labels.
    * `positions` a list whose kth element is the position of binary vector k after the automorphism.
    """

    def __init__(self, m: int, pairs: list[tuple[str, str]]):
        """Input:
        * `m` the bit count of the binary vector labels.
        * `pairs` a list of pairs of binary vector labels, each representing a swap.
        """
        self.M = m
        list_ = list(range(2**m))
        for label_1, label_2 in pairs:
            p1 = int(label_1, 2)
            p2 = int(label_2, 2)
            list_[p1], list_[p2] = list_[p2], list_[p1]
        self.positions = list_

    def __str__(self):
        return f"{list(range(2**self.M))} ->\n{self.positions}"

    @classmethod
    def from_positions(cls, m: int, positions: list[int]) -> 'Automorphism':
        new = cls.__new__(cls)
        new.M = m
        new.positions = positions
        return new

    def compose(self, other: 'Automorphism') -> 'Automorphism':
        if self.M != other.M:
            raise ValueError("Cannot compose automorphisms of different sizes")
        new_positions = [other.positions[p] for p in self.positions]
        return Automorphism.from_positions(self.M, new_positions)

    def update(self, pairs: list[tuple[str, str]]):
        list_ = self.positions
        for label_1, label_2 in pairs:
            p1 = int(label_1, 2)
            p2 = int(label_2, 2)
            list_[p1], list_[p2] = list_[p2], list_[p1]
    
    @property
    def matrix(self):
        """Represent as an n x n matrix."""
        matrix = np.zeros((self.M**2, self.M**2), dtype=int)
        for p, q in enumerate(self.positions):
            matrix[q, p] = 1
        return matrix
    
    def swap_type(self):
        swap_gates: set[tuple[int, int]] = set()
        encountered_qubits: set[int] = set()
        for row_index, position in enumerate(self.positions):
            if row_index not in encountered_qubits and row_index != position:
                swap_gates.add((row_index, position))
                encountered_qubits.add(position)
                encountered_qubits.add(row_index)
        return swap_gates
    
    def swap_type_circuit(self):
        swap_gates = self.swap_type()
        circuit = stim.Circuit()
        for q1, q2 in swap_gates:
            circuit.append_operation("SWAP", [q1, q2])
        return circuit
    
    def phase_type(self):
        cz_gates: set[tuple[int, int]] = set()
        encountered_qubits: set[int] = set()
        s_gates: set[int] = set()
        for row_index, position in enumerate(self.positions):
            if row_index == position:
                s_gates.add(row_index)
            elif row_index not in encountered_qubits:
                cz_gates.add((row_index, position))
                encountered_qubits.add(position)
                encountered_qubits.add(row_index)
        return cz_gates, s_gates
    
    def phase_type_circuit(self):
        cz_gates, s_gates = self.phase_type()
        circuit = stim.Circuit()
        for q1, q2 in cz_gates:
            circuit.append_operation("CZ", [q1, q2])
        circuit.append_operation("S", s_gates)
        return circuit


def rref_gf2(matrix):  # TODO: use galois package instead https://stackoverflow.com/questions/56856378/fast-computation-of-matrix-rank-over-gf2
    """Compute the reduced row echelon form of a binary matrix over GF(2)."""
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


if __name__ == "__main__":
    m = 6
    qrm = QuantumReedMuller(m)
    qrm.print()
    circuit = qrm.automorphism('Q', [(1, 2)]).phase_type_circuit()
    print(circuit.diagram())