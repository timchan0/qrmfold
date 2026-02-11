from collections.abc import Iterable
from typing import Literal
import itertools

import numpy as np
import stim


class Automorphism:
    """An automorphism of the classical Reed-Muller code RM(m/2-1, m)."""

    def __init__(self, num_variables: int, pairs: Iterable[tuple[str, str]]):
        """
        :param num_variables: The number m of variables of the Reed-Muller code.
        :param pairs: Iterable of distinct bit index pairs specifying swaps.
        """
        self.NUM_VARIABLES = num_variables
        """The number m of variables of the Reed-Muller code."""
        list_ = list(range(2**num_variables))
        for label_1, label_2 in pairs:
            p1 = int(label_1, 2)
            p2 = int(label_2, 2)
            list_[p1], list_[p2] = list_[p2], list_[p1]
        self.positions = list_
        """The permutation of the automorphism, represented as a length-2^m list
        such that ``positions[k]`` is the position of bit k after permutation.
        """

    def __str__(self):
        return f"{list(range(2**self.NUM_VARIABLES))} ->\n{self.positions}"

    def update(self, pairs: Iterable[tuple[str, str]]):
        list_ = self.positions
        for label_1, label_2 in pairs:
            p1 = int(label_1, 2)
            p2 = int(label_2, 2)
            list_[p1], list_[p2] = list_[p2], list_[p1]

    @property
    def matrix(self):
        """The 2^m x 2^m permutation matrix of the automorphism."""
        n = 2**self.NUM_VARIABLES
        out = np.zeros((n, n), dtype=np.uint8)
        for p, q in enumerate(self.positions):
            out[q, p] = 1
        return out

    def gate(self, gate_type: Literal['swap', 'phase']):
        """Return the physical circuit of this automorphism.

        :param gate_type: Gate type i.e. swap or phase.
        :returns circuit: A ``stim.Circuit`` of this automorphism.
        :raises ValueError: If ``gate_type`` is not recognized.
        """
        out = stim.Circuit()
        if gate_type == 'swap':
            swap_gates = self._swap_type()
            out.append("SWAP", itertools.chain.from_iterable(swap_gates), ())
        elif gate_type == 'phase':
            cz_gates, s_gates = self._phase_type()
            out.append("CZ", itertools.chain.from_iterable(cz_gates), ())
            out.append("S", s_gates, ())
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
        return out

    def _swap_type(self):
        """Return the SWAP-type gate of this automorphism.

        :returns swap_gates: A set of unordered pairs,
        each representing a SWAP gate between physical two qubits.
        """
        swap_gates: set[frozenset[int]] = set()
        for row_index, position in enumerate(self.positions):
            if row_index < position:
                swap_gates.add(frozenset({row_index, position}))
        return swap_gates

    def _phase_type(self):
        """Return the phase-type gate of this automorphism.

        :returns cz_gates: A set of unordered pairs representing physical CZ gates.
        :returns s_gates: A set of physical qubit indices where an S gate is applied.
        """
        cz_gates: set[frozenset[int]] = set()
        s_gates: set[int] = set()
        for row_index, position in enumerate(self.positions):
            if row_index == position:
                s_gates.add(row_index)
            elif row_index < position:
                cz_gates.add(frozenset({row_index, position}))
        return cz_gates, s_gates