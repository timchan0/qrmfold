from collections.abc import Collection, Iterable, Sequence
import itertools
from functools import reduce
import math
from typing import Literal

import numpy as np
from reedmuller.reedmuller import _vector_mult
import stim

from qrmfold import logical_index_to_subset_maps
from qrmfold._automorphism import Automorphism
from qrmfold.utils import all_bitstrings, extract_arguments, powerset, sign_to_power, sliding_window


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
        NOTE: we must have 0 <= i < m.
        """
        return ([0] * 2**i + [1] * 2**i) * 2 ** (self.M-i-1)


def _get_residual_slices(m: int, i: int, j: int):
    smaller, larger = min(i, j), max(i, j)
    residual_1 = all_bitstrings(m-larger)
    residual_2 = all_bitstrings(larger-1-smaller)
    residual_3 = all_bitstrings(smaller-1)
    return itertools.product(residual_1, residual_2, residual_3)


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


def _get_intermediate_subsets(b_subset: set[int], b_prime_subset: set[int]):
    """Return a list of subsets `list_` such that
    each subset in `[b_subset] + list_ + [b_prime_subset]`
    differs from the previous one by exactly one basis vector.

    Require: `b_subset` and `b_prime_subset` differ by at least one integer.
    """
    b_take_b_prime = b_subset.difference(b_prime_subset)
    b_prime_take_b = b_prime_subset.difference(b_subset)
    try:
        b_take_b_prime.pop()
        b_prime_take_b.pop()
    except KeyError:
        raise ValueError("b_subset and b_prime_subset must differ by at least one basis vector.")
    intermediate_subsets: list[set[int]] = []
    current_subset = b_subset.copy()
    for i, j in zip(b_take_b_prime, b_prime_take_b, strict=True):
        current_subset.remove(i)
        current_subset.add(j)
        intermediate_subsets.append(current_subset.copy())
    return intermediate_subsets


_signature_to_pauli = {
    (False, False): '_',
    (False, True): 'X',
    (True, True): 'Y',
    (True, False): 'Z',
}
"""Map from anticommutation with (X, Z) to the unique Pauli operator with that signature."""


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
        self._validate_init_inputs(m, logical_index_to_subset)
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
        self.logical_operators: dict[tuple[int, ...], tuple[stim.PauliString, stim.PauliString]] = {}
        for subset, support in self.logical_x_supports.items():
            x = stim.PauliString('X' if p else 'I' for p in support)
            complement = tuple(k for k in range(1, m+1) if k not in subset)
            z = stim.PauliString('Z' if p else 'I' for p in self.logical_x_supports[complement])
            self.logical_operators[subset] = (x, z)
        self.logical_index_to_subset = logical_index_to_subset_maps.default(m) \
            if logical_index_to_subset is None else logical_index_to_subset
        self.subset_to_logical_index = {
            frozenset(subset): index for index, subset in self.logical_index_to_subset.items()
        }

    @staticmethod
    def _validate_init_inputs(m: int, logical_index_to_subset: None | dict[int, tuple[int, ...]]):
        """Check that `m` is even and that `logical_index_to_subset` has the correct codomain."""
        if m % 2:
            raise ValueError("m must be even")
        if logical_index_to_subset is None:
            return
        target: set[frozenset[int]] = {frozenset(subset) for subset in itertools.combinations(range(1, m+1), m//2)}
        actual: set[frozenset[int]] = {frozenset(subset) for subset in logical_index_to_subset.values()}
        if target != actual:
            raise ValueError("self.logical_index_to_subset must be a 1-to-1 map from logical qubit index to subset of [m] of cardinality m/2.")

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

    def q_automorphism_phase_type_product(self, pairs: Collection[tuple[int, int]]):
        """Return the physical circuit of the product_{L subset of K} U_P(Q(L)), where K is the set of pairs."""
        return sum((
            self.automorphism('Q', l_subset).phase_type_circuit() for l_subset in powerset(pairs)
        ), start=stim.Circuit())

    def q_automorphism_phase_type_logical_action(self, pairs: Collection[tuple[int, int]]):
        """Return the logical action of the U_P(Q(K)) where K is the set of pairs."""
        circuit = stim.Circuit()
        circuit.append('I', self.logical_index_to_subset.keys(), ())
        for l_subset in powerset(pairs, self.M//2 - 2):
            self._logical_action_helper(circuit, l_subset, gates=['CZ'])
        if len(pairs) >= self.M//2 - 1:
            for l_subset in itertools.combinations(pairs, self.M//2 - 1):
                self._logical_action_helper(circuit, l_subset, gates=['CZ', 'Z'])
        if len(pairs) == self.M//2:
            arguments_0 = extract_arguments(0, pairs)
            logical_index = self.subset_to_logical_index[frozenset(arguments_0)]
            gate = 'S_DAG' if self.M//2 % 2 else 'S'
            circuit.append(gate, logical_index, ())
        return circuit

    def q_automorphism_phase_type_product_logical_action(self, pairs: Collection[tuple[int, int]]):
        """Return the logical action of the product_{L subset of K} U_P(Q(L)), where K is the set of pairs."""
        circuit = stim.Circuit()
        circuit.append('I', self.logical_index_to_subset.keys(), ())
        if len(pairs) <= self.M//2 - 2:
            self._logical_action_helper(circuit, pairs, gates=['CZ'])
        elif len(pairs) == self.M//2 - 1:
            self._logical_action_helper(circuit, pairs, gates=['CZ', 'Z'])
        elif len(pairs) == self.M//2:
            arguments_0 = extract_arguments(0, pairs)
            logical_index = self.subset_to_logical_index[frozenset(arguments_0)]
            gate = 'S_DAG' if self.M//2 % 2 else 'S'
            circuit.append(gate, logical_index, ())
        else:
            raise ValueError("pairs length cannot exceed m/2")
        return circuit

    def logical(
            self,
            gate: Literal['S', 'H', 'CZ_XX', 'SWAP'],
            logical_indices: Iterable[int],
    ):
        """Return the physical circuit inducing logical `gate` on the given logical qubit index/indices."""
        if gate in {'S', 'H'}:
            try:
                logical_index, = logical_indices
            except ValueError:
                raise ValueError(f"Logical gate {gate} requires exactly one logical index.")
            b_subset = set(self.logical_index_to_subset[logical_index])
            if gate == 'S':
                return self._logical_s(b_subset)
            return self._logical_h(b_subset)
        try:
            logical_index_0, logical_index_1 = logical_indices
        except ValueError:
            raise ValueError(f"Logical gate {gate} requires exactly two logical indices.")
        restricted_gate = self._logical_swap_restricted if gate == 'SWAP' else self._logical_czxx_restricted
        b_subset = set(self.logical_index_to_subset[logical_index_0])
        b_prime_subset = set(self.logical_index_to_subset[logical_index_1])
        intermediate_subsets = _get_intermediate_subsets(b_subset, b_prime_subset)
        if not intermediate_subsets:
            return restricted_gate(b_subset, b_prime_subset)
        entry = sum((
            self._logical_swap_restricted(a, b) for a, b in
            sliding_window([b_subset] + intermediate_subsets, 2)
        ), start=stim.Circuit())
        apex = restricted_gate(intermediate_subsets[-1], b_prime_subset)
        return entry + apex + entry.inverse()

    def _logical_s(self, b_subset: set[int]):
        pairs = list(zip(b_subset, self._complement(b_subset), strict=True))
        physical_circuit = self.q_automorphism_phase_type_product(pairs)
        if self.M//2 % 2:
            return physical_circuit.inverse()
        return physical_circuit

    def _logical_h(self, b_subset: set[int]):
        s_b = self._logical_s(b_subset)
        transversal_h = stim.Circuit(f'H {' '.join(str(k) for k in range(s_b.num_qubits))}')
        s_b_complement = self._logical_s(self._complement(b_subset))
        return s_b + transversal_h + s_b_complement + transversal_h + s_b

    def _logical_czxx_restricted(self, b_subset: set[int], b_prime_subset: set[int]):
        """Return the physical circuit inducing logical CZ_XX on the given logical qubits,
        where the logical qubits are labelled by subsets that differ by exactly one basis vector.
        """
        arguments_0 = b_subset.intersection(b_prime_subset)
        if len(arguments_0) != self.M//2 - 1:
            raise ValueError(f"Logical qubit indices {b_subset}, {b_prime_subset} must differ by exactly one basis vector.")
        arguments_1 = self._complement(b_subset.union(b_prime_subset))
        pairs = list(zip(arguments_0, arguments_1, strict=True))
        return self.q_automorphism_phase_type_product(pairs)

    def _logical_swap_restricted(self, b_subset: set[int], b_prime_subset: set[int]):
        """Return the physical circuit inducing logical SWAP on the given logical qubits,
        where the logical qubits are labelled by subsets that differ by exactly one basis vector.
        """
        czxx = self._logical_czxx_restricted(b_subset, b_prime_subset)
        hh = self._logical_h(b_subset) + self._logical_h(b_prime_subset)
        return 3 * (czxx + hh)

    def _logical_action_helper(
            self,
            circuit: stim.Circuit,
            pairs: Collection[tuple[int, int]],
            gates: Sequence[str],
    ):
        """Append gates in `gates` to `circuit` according to `pairs` whose length is <m/2."""
        arguments_0 = extract_arguments(0, pairs)
        arguments_1 = extract_arguments(1, pairs)
        encountered_qubits: set[int] = set()
        for logical_index, b_subset in self.logical_index_to_subset.items():
            if logical_index not in encountered_qubits and arguments_0.issubset(b_subset) and arguments_1.isdisjoint(b_subset):
                b_complement = self._complement(b_subset)
                b_prime_subset = b_complement.union(arguments_0).difference(arguments_1)
                logical_index_prime = self.subset_to_logical_index[frozenset(b_prime_subset)]
                for gate in gates:
                    circuit.append(gate, [logical_index, logical_index_prime], ())
                encountered_qubits.add(logical_index_prime)

    def _complement(self, subset: Collection[int]):
        """Return the complement of `subset` of [m]."""
        return set(range(1, self.M+1)).difference(subset)

    def _get_logical_tableau(self, physical_circuit: stim.Circuit):
        """Get the logical tableau induced by `physical_circuit`.

        Require:
        * `physical_circuit` preserves the stabilizer group.
        """
        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []
        for _, subset in sorted(self.logical_index_to_subset.items(), key=lambda item: item[0]):
            for logical_operator, conjugated_generator_list in zip(
                self.logical_operators[subset],
                (xs, zs),
                strict=True,
            ):
                transformed = logical_operator.after(physical_circuit)
                transformed_logical_action = ''.join(_signature_to_pauli[tuple(
                    not transformed.commutes(observable) for observable in self.logical_operators[_subset]
                )] for _, _subset in sorted(self.logical_index_to_subset.items(), key=lambda item: item[0])) # type: ignore

                # construct logical representative
                logical_representative = stim.PauliString()
                for (_, _subset), pauli in zip(
                    sorted(self.logical_index_to_subset.items(), key=lambda item: item[0]),
                    transformed_logical_action,
                    strict=True,
                ):
                    _x, _z = self.logical_operators[_subset]
                    if pauli == 'X':
                        logical_representative *= _x
                    elif pauli == 'Y':
                        logical_representative *= 1j * _x * _z
                    elif pauli == 'Z':
                        logical_representative *= _z

                phase_stabilizer = transformed * logical_representative
                # extract phase
                phase_exponent = (sign_to_power[phase_stabilizer.sign] + \
                    len(phase_stabilizer.pauli_indices('Y'))) % 4

                conjugated_generator_list.append(1j**phase_exponent * stim.PauliString(transformed_logical_action))
        tableau = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
        return tableau