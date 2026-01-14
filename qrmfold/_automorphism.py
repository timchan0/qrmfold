import numpy as np
import stim


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