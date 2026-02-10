from collections.abc import Collection, Iterable, Sequence
import itertools
from functools import cached_property
import math
from typing import Literal

import numpy as np
from reedmuller.reedmuller import _vector_mult, _vector_neg
import stim

from qrmfold import logical_qubit_orderings
from qrmfold._automorphism import Automorphism
from qrmfold._depth_reducer import DepthReducer
from qrmfold.utils import all_bitstrings, complement, extract_arguments, powerset, sign_to_power, rref_gf2


class ReedMuller:
    """The classical Reed-Muller code RM(r, m)."""

    def __init__(self, order: int, bit_count: int, minimize_weight: bool = True):
        """
        :param order: The order r of the Reed-Muller code.
        :param bit_count: The number m of bits of the Reed-Muller code.
        :param minimize_weight: Whether to minimize the weight
            of each generator to 2^{m/2 + 1}.
        """
        self.ORDER = order
        """The order r of the Reed-Muller code."""
        self.BIT_COUNT = bit_count
        """The number m of bits of the Reed-Muller code."""
        self.BASIS = tuple(self._construct_basis_vector(i) for i in range(bit_count))
        """Basis vectors {v_i : 0 <= i < m} for the Reed-Muller code of m bits."""

        _all_ones = [1] * 2**bit_count
        _generator_matrix: list[list[int]] = []
        _basis_neg = tuple(_vector_neg(v) for v in self.BASIS)
        for cardinality in range(order + 1):
            for subset in itertools.combinations(range(bit_count), cardinality):
                vector_subset = [self.BASIS[i] for i in subset]
                generator: list[int] = _all_ones if not subset else _vector_mult(*vector_subset)
                
                if minimize_weight:
                    c = bit_count//2 - 1 - cardinality
                    _complement = complement(bit_count, subset, start=0)
                    additional_vectors = [_basis_neg[i] for i in _complement][:c]
                    generator = _vector_mult(generator, *additional_vectors)
                
                _generator_matrix.append(generator)
        self.generator_matrix = _generator_matrix
        """Matrix with a row for every subset of the basis vectors
        of cardinality at most the order r of the code.
        """

    def _construct_basis_vector(self, index: int):
        """Construct the ith basis vector of the Reed-Muller code.

        The vector is a string of 2^i zeros followed by 2^i ones repeated 2^(m - i - 1) times.

        :param index: Basis index i satisfying 0 <= i < m,
        where m is the bit count of the Reed-Muller code.
        :returns basis_vector: The ith basis vector of length 2^m.
        """
        list_: list[int] = ([0] * 2**index + [1] * 2**index) * 2 ** (self.BIT_COUNT-index-1)
        return tuple(list_)


def _get_residual_slices(bit_count: int, i: int, j: int):
    """Generate residual bitstring slices used by the automorphisms.

    :param bit_count: Bit count of the bit index.
    :param i: First basis vector index (1-indexed).
    :param j: Second basis vector index (1-indexed).
    :returns residual_slices: An iterator over triples of residual bitstrings.
    """
    smaller, larger = min(i, j), max(i, j)
    residual_1 = all_bitstrings(bit_count-larger)
    residual_2 = all_bitstrings(larger-1-smaller)
    residual_3 = all_bitstrings(smaller-1)
    return itertools.product(residual_1, residual_2, residual_3)


def _p_automorphism(bit_count: int, i: int, j: int):
    """Swap basis vectors v_i and v_j (1-indexed).

    :param bit_count: Bit count of the bit index.
    :param i: First basis vector index (1-indexed).
    :param j: Second basis vector index (1-indexed).
    :returns pairs: A list of swapped bit index pairs ``(s1, s2)``.
        For each pair, ``s1`` has a 0 in position ``i`` and a 1 in position ``j``,
        and ``s2`` has a 1 in position ``i`` and a 0 in position ``j``.
    """
    result: list[tuple[str, str]] = []
    for a, b, c in _get_residual_slices(bit_count, i, j):
        s1 = a + '0' + b + '1' + c
        s2 = a + '1' + b + '0' + c
        result.append((s1, s2))
    return result


def _q_automorphism(bit_count: int, i: int, j: int):
    """Add basis vector v_j onto basis vector v_i (1-indexed).

    :param bit_count: Bit count of the bit index.
    :param i: Target basis vector index (1-indexed).
    :param j: Source basis vector index (1-indexed).
    :returns pairs: A list of swapped bit index pairs ``(s1, s2)``.
        For each pair, ``s1`` has a 0 in position ``i`` and a 1 in position ``j``,
        and ``s2`` has a 1 in position ``i`` and a 1 in position ``j``.
    """
    if i < j:
        s1_larger, s1_smaller = '1', '0'
    else:
        s1_larger, s1_smaller = '0', '1'
    result: list[tuple[str, str]] = []
    for a, b, c in _get_residual_slices(bit_count, i, j):
        s1 = a + s1_larger + b + s1_smaller + c
        s2 = a + '1' + b + '1' + c
        result.append((s1, s2))
    return result


def _get_intermediate_subsets(b_subset: set[int], b_prime_subset: set[int]):
    """Compute intermediate subsets between two subsets B and B'.

    :param b_subset: Starting subset B.
    :param b_prime_subset: Ending subset B'.
    :returns intermediate_subsets: A list such that each subset in
        ``(b_subset, *intermediate_subsets, b_prime_subset)``
        differs from the previous one by exactly one element.
    :raises ValueError: If B and B' do not differ by
        at least one element.
    """
    b_take_b_prime = b_subset.difference(b_prime_subset)
    b_prime_take_b = b_prime_subset.difference(b_subset)
    try:
        b_take_b_prime.pop()
        b_prime_take_b.pop()
    except KeyError:
        raise ValueError("b_subset and b_prime_subset must differ by at least one element.")
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
    """The quantum Reed-Muller code CSS(RM(m/2 - 1, m), RM(m/2 - 1, m))."""

    def __init__(
            self,
            bit_count: int,
            minimize_weight: bool = True,
            logical_qubit_ordering: None | dict[int, set[int]] = None,
    ):
        """
        :param bit_count: The number m of bits of the
            underlying classical Reed-Muller code; must be even.
        :param minimize_weight: Whether to minimize the weight
            of each stabilizer generator to 2^{m/2 + 1}.
        :param logical_qubit_ordering: Optional 1-to-1 map from logical qubit
            index to a subset of [m] of cardinality m/2.
            If not specified, subsets are ordered lexicographically.
        :raises ValueError: If ``bit_count`` is odd,
        or if ``logical_qubit_ordering`` does not have the required codomain.
        """
        self._validate_init_inputs(bit_count, logical_qubit_ordering)
        self.BIT_COUNT = bit_count
        """The number m of bits of the underlying classical Reed-Muller code."""
        order = bit_count//2 - 1
        self.classical = ReedMuller(order, bit_count, minimize_weight=minimize_weight)
        """The underlying classical Reed-Muller code RM(m/2 - 1, m)."""
        self.stabilizer_generators = {
            pauli: [stim.PauliString(pauli if p else 'I' for p in g) for g in self.classical.generator_matrix]
            for pauli in ('X', 'Z')
        }
        """Map from Pauli (X or Z) to a list of stabilizer generators of that type."""
        self._logical_x_supports: dict[frozenset[int], list[int]] = {
            frozenset(subset): _vector_mult(*(self.classical.BASIS[i-1] for i in subset))
            for subset in itertools.combinations(range(1, bit_count+1), bit_count//2)
        }
        self.logical_operators: dict[frozenset[int], tuple[stim.PauliString, stim.PauliString]] = {}
        """Map from logical qubit (labelled by a subset of [m]) to its X and Z logical operators."""
        for subset, support in self._logical_x_supports.items():
            x = stim.PauliString('X' if p else 'I' for p in support)
            _complement = frozenset(complement(bit_count, subset))
            z = stim.PauliString('Z' if p else 'I' for p in self._logical_x_supports[_complement])
            self.logical_operators[subset] = (x, z)
        self.logical_qubit_ordering = logical_qubit_orderings.lexicographic(bit_count) \
            if logical_qubit_ordering is None else logical_qubit_ordering
        """Map from logical qubit index to a unique subset of [m] of cardinality m/2."""
        self.subset_to_logical_index = {
            frozenset(subset): index for index, subset in self.logical_qubit_ordering.items()
        }
        """Map from subset of [m] of cardinality m/2 to logical qubit index."""

    @staticmethod
    def _validate_init_inputs(bit_count: int, logical_qubit_ordering: None | dict[int, set[int]]):
        """Validate constructor inputs.

        :param bit_count: The number m of bits of the
            underlying classical Reed-Muller code; must be even.
        :param logical_qubit_ordering: Candidate map from logical index to subset of [m] of cardinality m/2.
        :raises ValueError: If ``bit_count`` is odd,
        or if ``logical_qubit_ordering`` does not have the required codomain.
        """
        if bit_count % 2:
            raise ValueError("bit_count must be even")
        if logical_qubit_ordering is None:
            return
        target: set[frozenset[int]] = {frozenset(subset) for subset in itertools.combinations(range(1, bit_count+1), bit_count//2)}
        actual: set[frozenset[int]] = {frozenset(subset) for subset in logical_qubit_ordering.values()}
        if target != actual:
            raise ValueError("logical_qubit_ordering must be a 1-to-1 map from logical qubit index to subset of [m] of cardinality m/2.")

    def print(self):
        """Print the stabilizer generators and logical operators."""
        for pauli in ('X', 'Z'):
            print(f'{pauli} stabilizer generators:')
            for g in self.stabilizer_generators[pauli]:
                print(g)
        print('logical qubits and their X and Z operators:')
        digit_count = math.ceil(math.log10(len(self._logical_x_supports)))
        for logical_index, subset in self.logical_qubit_ordering.items():
            print(
                f'{logical_index:{digit_count}d}',
                subset,
                *(str(pauli) for pauli in self.logical_operators[frozenset(subset)]),
            )

    @cached_property
    def stabilizer_generators_rref(self):
        """The stabilizer generators as a binary symplectic matrix in reduced row echelon form."""
        stabilizers_bsf: list[np.ndarray[tuple[int], np.dtype[np.bool_]]] = []
        for basis_generators in self.stabilizer_generators.values():
            for generator in basis_generators:
                xs, zs = generator.to_numpy()
                stabilizers_bsf.append(np.append(xs, zs))
        return rref_gf2(stabilizers_bsf)

    def _automorphism(
            self,
            automorphism_type: Literal['trivial', 'P', 'Q'] = 'trivial',
            pairs: None | Iterable[tuple[int, int]] = None,
    ):
        """Construct an automorphism of the underlying classical Reed-Muller code.

        :param automorphism_type: The automorphism type: trivial, P, or Q.
            The trivial automorphism maps every element to itself.
            P(i, j) swaps basis vectors v_i and v_j.
            Q(i, j) adds basis vector v_j onto basis vector v_i.
        :param pairs: Iterable of integer pairs (i, j) satisfying
            1 <= i, j <= m, where m is the bit count of the quantum Reed-Muller code.
            Each integer in ``pairs`` must be distinct.
            If omitted (or if ``automorphism_type == 'trivial'``),
            the trivial automorphism is returned.
        :returns automorphism: A :class:`qrmfold._automorphism.Automorphism` object.
        :raises ValueError: If any index in ``pairs`` is out of range or is repeated,
            or if ``automorphism_type`` is not recognized.
        """
        automorphism = Automorphism(self.BIT_COUNT, ())
        if pairs is not None and automorphism_type != 'trivial':
            exhausted_indices: set[int] = set()
            for i, j in pairs:
                if not ((1 <= i <= self.BIT_COUNT) and (1 <= j <= self.BIT_COUNT)):
                    raise ValueError(f"i and j must be between 1 and {self.BIT_COUNT}, inclusive")
                if (pair:={i, j}) & exhausted_indices:
                    raise ValueError("Each integer in pairs must be distinct")
                exhausted_indices.update(pair)
                if automorphism_type == 'P':
                    automorphism.update(_p_automorphism(self.BIT_COUNT, i, j))
                elif automorphism_type == 'Q':
                    automorphism.update(_q_automorphism(self.BIT_COUNT, i, j))
                else:
                    raise ValueError("automorphism_type must be 'trivial', 'P', or 'Q'")
        return automorphism

    def automorphism(
            self,
            pairs: None | Collection[tuple[int, int]] = None,
            automorphism_type: Literal['trivial', 'P', 'Q'] = 'Q',
            gate_type: Literal['swap', 'phase'] = 'phase',
    ):
        """Return the physical circuit of U_t(a(K)).

        :param pairs: Set K of integer pairs (i, j) satisfying
            1 <= i, j <= m, where m is the bit count of the quantum Reed-Muller code.
            Each integer in ``pairs`` must be distinct.
            If omitted, treated as empty.
        :param automorphism_type: The automorphism type a: trivial, P, or Q.
            The trivial automorphism maps every element to itself.
            P(i, j) swaps basis vectors v_i and v_j.
            Q(i, j) adds basis vector v_j onto basis vector v_i.
        :param gate_type: Gate type t: swap or phase.
        :returns circuit: A ``stim.Circuit`` representing the physical circuit of U_t(a(K)).
        """
        return self._automorphism(automorphism_type=automorphism_type, pairs=pairs).gate(gate_type)
    
    def automorphism_product(
            self,
            pairs: None | Collection[tuple[int, int]] = None,
            automorphism_type: Literal['trivial', 'P', 'Q'] = 'Q',
            gate_type: Literal['swap', 'phase'] = 'phase',
    ):
        """Return the physical circuit of ``\\prod_{L \\subseteq K} U_t(a(L))``.

        :param pairs: Set K of integer pairs (i, j) satisfying
            1 <= i, j <= m, where m is the bit count of the quantum Reed-Muller code.
            Each integer in ``pairs`` must be distinct.
            If omitted, treated as empty.
        :param automorphism_type: The automorphism type a: trivial, P, or Q.
            The trivial automorphism maps every element to itself.
            P(i, j) swaps basis vectors v_i and v_j.
            Q(i, j) adds basis vector v_j onto basis vector v_i.
        :param gate_type: Gate type t: swap or phase.
        :returns circuit: A ``stim.Circuit`` representing the physical circuit of the product.
        """
        if pairs is None:
            pairs = []
        return sum((self.automorphism(
            pairs=l_subset,
            automorphism_type=automorphism_type,
            gate_type=gate_type,
        ) for l_subset in powerset(pairs)), start=stim.Circuit())

    def q_phase_logical_action(self, pairs: Collection[tuple[int, int]]):
        """Compute the logical action of U_P(Q(K)).

        :param pairs: Set K of pairs.
        :returns circuit: A ``stim.Circuit`` acting on the logical qubits of the code.
        """
        circuit = self._logical_action_starter()
        for l_subset in powerset(pairs, self.BIT_COUNT//2 - 2):
            self._logical_action_helper(circuit, l_subset, gates=['CZ'])
        if len(pairs) >= self.BIT_COUNT//2 - 1:
            for l_subset in itertools.combinations(pairs, self.BIT_COUNT//2 - 1):
                self._logical_action_helper(circuit, l_subset, gates=['CZ', 'Z'])
        if len(pairs) == self.BIT_COUNT//2:
            arguments_0 = extract_arguments(0, pairs)
            logical_index = self.subset_to_logical_index[frozenset(arguments_0)]
            gate = 'S_DAG' if self.BIT_COUNT//2 % 2 else 'S'
            circuit.append(gate, logical_index, ())
        return circuit

    def q_phase_product_logical_action(self, pairs: Collection[tuple[int, int]]):
        """Compute the logical action of ``\\prod_{L \\subseteq K} U_P(Q(L))``.

        :param pairs: Set K of pairs.
        :returns circuit: A ``stim.Circuit`` acting on the logical qubits of the code.
        :raises ValueError: If ``pairs`` longer than ``self.BIT_COUNT/2``.
        """
        circuit = self._logical_action_starter()
        if len(pairs) <= self.BIT_COUNT//2 - 2:
            self._logical_action_helper(circuit, pairs, gates=['CZ'])
        elif len(pairs) == self.BIT_COUNT//2 - 1:
            self._logical_action_helper(circuit, pairs, gates=['CZ', 'Z'])
        elif len(pairs) == self.BIT_COUNT//2:
            arguments_0 = extract_arguments(0, pairs)
            logical_index = self.subset_to_logical_index[frozenset(arguments_0)]
            gate = 'S_DAG' if self.BIT_COUNT//2 % 2 else 'S'
            circuit.append(gate, logical_index, ())
        else:
            raise ValueError("pairs cannot be longer than self.BIT_COUNT/2")
        return circuit

    def _logical_action_starter(self):
        circuit = stim.Circuit()
        circuit.append('I', self.logical_qubit_ordering.keys(), ())
        return circuit

    def _logical_action_helper(
            self,
            circuit: stim.Circuit,
            pairs: Collection[tuple[int, int]],
            gates: Sequence[str],
    ):
        """Helper for :meth:`q_phase_logical_action` and :meth:`q_phase_product_logical_action`.

        :param circuit: Circuit to append to.
        :param pairs: Pair set (see caller) of length <m/2.
        :param gates: Gate names to append for each affected logical qubit pair.
        :returns None: Mutates ``circuit`` in-place.
        """
        arguments_0 = extract_arguments(0, pairs)
        arguments_1 = extract_arguments(1, pairs)
        encountered_qubits: set[int] = set()
        for logical_index, b_subset in self.logical_qubit_ordering.items():
            if logical_index not in encountered_qubits and arguments_0.issubset(b_subset) and arguments_1.isdisjoint(b_subset):
                b_complement = complement(self.BIT_COUNT, b_subset)
                b_prime_subset = b_complement.union(arguments_0).difference(arguments_1)
                logical_index_prime = self.subset_to_logical_index[frozenset(b_prime_subset)]
                for gate in gates:
                    circuit.append(gate, [logical_index, logical_index_prime], ())
                encountered_qubits.add(logical_index_prime)

    def gate(
            self,
            name: Literal['S', 'H', 'ZZCZ', 'SWAP'],
            targets: Iterable[int],
            reduce_depth: bool = True,
    ):
        """Build the physical circuit inducing a logical gate.

        :param name: The name of the logical gate to implement.
            This can be S, H, ZZCZ, or SWAP.
        :param targets: Logical qubit indices to apply the gate on.
            For 2-qubit gates, the operation is broadcast over
            consecutive target pairs, so an even target count is required.
        :param reduce_depth: Whether to apply basic depth reduction before returning.
        :returns circuit: A ``stim.Circuit`` inducing the requested logical action.
        :raises ValueError: If a 2-qubit gate is requested with an odd number of targets.
        """
        if name == 'S' or name == 'H':
            _gate = self._s if name == 'S' else self._h
            out = sum((_gate(
                set(self.logical_qubit_ordering[target])
            ) for target in targets), start=stim.Circuit())
        else:
            out = stim.Circuit()
            try:
                for target_0, target_1 in itertools.batched(targets, 2, strict=True):
                    out += self._2_qubit_gate(target_0, target_1, name)
            except ValueError:
                raise ValueError(f"2-qubit gate {name} requires an even target count but was given {targets}.")
        if reduce_depth:
            return DepthReducer.reduce(out)
        return out

    def _s(self, b_subset: set[int]):
        """Build the physical circuit inducing a logical S.

        :param b_subset: Logical qubit label B as a subset of [m] of cardinality m/2.
        :returns circuit: A ``stim.Circuit`` inducing the logical S on that qubit.
        """
        pairs = list(zip(b_subset, complement(self.BIT_COUNT, b_subset), strict=True))
        out = self.automorphism_product(pairs, automorphism_type='Q', gate_type='phase')
        if self.BIT_COUNT//2 % 2:
            return out.inverse()
        return out

    def _h(self, b_subset: set[int]):
        """Build the physical circuit inducing a logical H.

        :param b_subset: Logical qubit label B as a subset of [m] of cardinality m/2.
        :returns circuit: A ``stim.Circuit`` inducing the logical H on that qubit.
        """
        s_b = self._s(b_subset)
        transversal_h = stim.Circuit(f'H {' '.join(str(k) for k in range(s_b.num_qubits))}')
        s_b_complement = self._s(complement(self.BIT_COUNT, b_subset))
        return s_b + transversal_h + s_b_complement + transversal_h + s_b
    
    def _2_qubit_gate(self, target_0: int, target_1: int, name: Literal['SWAP', 'ZZCZ']):
        """Build a physical circuit implementing a 2-qubit logical gate.

        :param target_0: First logical qubit index.
        :param target_1: Second logical qubit index.
        :param name: 2-qubit logical gate: SWAP or ZZCZ.
        :returns circuit: A ``stim.Circuit`` inducing the requested logical action.
        """
        _gate = self._swap_restricted if name == 'SWAP' else self._zzcz_restricted
        b_subset = set(self.logical_qubit_ordering[target_0])
        b_prime_subset = set(self.logical_qubit_ordering[target_1])
        intermediate_subsets = _get_intermediate_subsets(b_subset, b_prime_subset)
        if not intermediate_subsets:
            return _gate(b_subset, b_prime_subset)
        entry = sum((
            self._swap_restricted(a, b) for a, b in
            itertools.pairwise([b_subset] + intermediate_subsets)
        ), start=stim.Circuit())
        apex = _gate(intermediate_subsets[-1], b_prime_subset)
        return entry + apex + entry.inverse()

    def _zzcz_restricted(self, b_subset: set[int], b_prime_subset: set[int]):
        """Build the physical circuit inducing a logical (ZZ)CZ.

        The two logical qubits must be labelled by subsets that differ by
        exactly one element.

        :param b_subset: First logical label B as a subset of [m] of cardinality m/2.
        :param b_prime_subset: Second logical label B'.
        :returns circuit: A ``stim.Circuit`` implementing logical (ZZ)CZ on B and B'.
        :raises ValueError: If B and B' do not differ by exactly one element.
        """
        arguments_0 = b_subset.intersection(b_prime_subset)
        if len(arguments_0) != self.BIT_COUNT//2 - 1:
            raise ValueError(f"Logical qubit labels {b_subset}, {b_prime_subset} must differ by exactly one element.")
        arguments_1 = complement(self.BIT_COUNT, b_subset.union(b_prime_subset))
        pairs = list(zip(arguments_0, arguments_1, strict=True))
        return self.automorphism_product(pairs, automorphism_type='Q', gate_type='phase')

    def _swap_restricted(self, b_subset: set[int], b_prime_subset: set[int]):
        """Build the physical circuit inducing a logical SWAP.

        The two logical qubits must be labelled by subsets that differ by
        exactly one element.
        
        :param b_subset: First logical label B as a subset of [m] of cardinality m/2.
        :param b_prime_subset: Second logical label B'.
        :returns circuit: A ``stim.Circuit`` implementing logical SWAP on B and B'.
        :raises ValueError: If B and B' do not differ by exactly one element.
        """
        zzcz = self._zzcz_restricted(b_subset, b_prime_subset)
        hh = self._h(b_subset) + self._h(b_prime_subset)
        return 3 * (zzcz + hh)

    def _get_logical_tableau(self, physical_circuit: stim.Circuit):
        """Compute the logical tableau induced by a physical circuit.

        :param physical_circuit: A physical circuit that preserve the
            stabilizer group.
        :returns tableau: The induced logical tableau.
        """
        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []
        for _, subset in sorted(self.logical_qubit_ordering.items(), key=lambda item: item[0]):
            for logical_operator, conjugated_generator_list in zip(
                self.logical_operators[frozenset(subset)],
                (xs, zs),
                strict=True,
            ):
                transformed = logical_operator.after(physical_circuit)
                transformed_logical_action = ''.join(_signature_to_pauli[tuple(
                    not transformed.commutes(observable) for observable in self.logical_operators[frozenset(_subset)]
                )] for _, _subset in sorted(self.logical_qubit_ordering.items(), key=lambda item: item[0])) # type: ignore

                # construct logical representative
                logical_representative = stim.PauliString()
                for (_, _subset), pauli in zip(
                    sorted(self.logical_qubit_ordering.items(), key=lambda item: item[0]),
                    transformed_logical_action,
                    strict=True,
                ):
                    _x, _z = self.logical_operators[frozenset(_subset)]
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