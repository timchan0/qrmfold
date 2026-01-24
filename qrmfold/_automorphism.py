from typing import Literal
import itertools

import numpy as np
import stim


class Automorphism:
    """An automorphism of ``2**m`` elements represented as a permutation.

    :ivar M: Bit count of the binary vector labels.
    :ivar positions: A list such that ``positions[k]`` is the position of binary
        vector ``k`` after applying the automorphism.
    """

    def __init__(self, m: int, pairs: list[tuple[str, str]]):
        """Create an automorphism from a list of swaps.

        :param m: Bit count of the binary vector labels.
        :param pairs: List of label pairs ``(label_1, label_2)`` (binary strings)
            specifying swaps.
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

    def update(self, pairs: list[tuple[str, str]]):
        list_ = self.positions
        for label_1, label_2 in pairs:
            p1 = int(label_1, 2)
            p2 = int(label_2, 2)
            list_[p1], list_[p2] = list_[p2], list_[p1]

    @property
    def matrix(self):
        """Represent the automorphism as a permutation matrix.

        :returns out: An n x n numpy array ``out`` such that applying the automorphism
            corresponds to multiplying by ``out``.
        """
        out = np.zeros((self.M**2, self.M**2), dtype=int)
        for p, q in enumerate(self.positions):
            out[q, p] = 1
        return out

    def gate(self, type_: Literal['swap', 'phase']):
        """Return the physical circuit for this automorphism.

        :param type_: Gate type (``'swap'`` or ``'phase'``).
        :returns: A ``stim.Circuit`` for this automorphism.
        :raises ValueError: If ``type_`` is not recognized.
        """
        out = stim.Circuit()
        if type_ == 'swap':
            swap_gates = self._swap_type()
            out.append("SWAP", itertools.chain.from_iterable(swap_gates), ())
        elif type_ == 'phase':
            cz_gates, s_gates = self._phase_type()
            out.append("CZ", itertools.chain.from_iterable(cz_gates), ())
            out.append("S", s_gates, ())
        else:
            raise ValueError(f"Unknown gate type: {type_}")
        return out

    def _swap_type(self):
        """Compute the SWAP-type gate for this automorphism.

        :returns: A set of unordered pairs, each representing a SWAP gate between
            two qubits.
        """
        swap_gates: set[frozenset[int]] = set()
        for row_index, position in enumerate(self.positions):
            if row_index < position:
                swap_gates.add(frozenset({row_index, position}))
        return swap_gates

    def _phase_type(self):
        """Compute the phase-type gate for this automorphism.

        :returns: A pair ``(cz_gates, s_gates)`` where ``cz_gates`` is a set of
            unordered pairs representing CZ gates, and ``s_gates`` is a set of
            qubit indices where an S gate is applied.
        """
        cz_gates: set[frozenset[int]] = set()
        s_gates: set[int] = set()
        for row_index, position in enumerate(self.positions):
            if row_index == position:
                s_gates.add(row_index)
            elif row_index < position:
                cz_gates.add(frozenset({row_index, position}))
        return cz_gates, s_gates