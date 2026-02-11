from collections import Counter, defaultdict

import stim


def reduce_circuit_depth(circuit: stim.Circuit):
    """Apply basic circuit depth reduction by accumulating commuting gates.

    Currently supports I, S, Z, S_DAG, CZ, H, CX,
    TICK, and repeat blocks.
    Accumulates the S, Z, S_DAG, and CZ gates
    and removes the I gates.

    :param circuit: Input circuit as a ``stim.Circuit``.
    :returns reduced_circuit: An equivalent circuit of reduced depth
        as a ``stim.Circuit``.
    :raises ValueError: If an unsupported instruction is encountered.
    """
    return _DepthReducer.reduce(circuit)


class _DepthReducer:

    _GATE_TO_EXPONENT = {'S': 1, 'Z': 2, 'S_DAG': 3}
    _PERIOD = 4

    def __init__(self) -> None:
        self.s_gates: Counter[int] = Counter()
        """Map from qubit index to the integer exponent of the S gate applied."""
        self.cz_to_moment: dict[frozenset[int], int] = {}
        """Map from an unordered pair of qubit indices to the timeslice of its CZ gate."""
        self.moment_to_cz: defaultdict[int, dict[int, int]] = defaultdict(dict)
        """Map from timeslice to a bidirectional map between qubit indices."""
    
    @classmethod
    def reduce(cls, circuit: stim.Circuit) -> stim.Circuit:
        """See `reduce_circuit_depth`."""
        out = stim.Circuit()
        reducer = cls()
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                out += reducer._to_circuit()
                reducer = cls()
                out += instruction.repeat_count * cls.reduce(instruction.body_copy())
            elif instruction.name in {'TICK', 'H', 'CX'}:
                out += reducer._to_circuit()
                reducer = cls()
                out.append(instruction)
            elif instruction.name == 'CZ':
                reducer._add_cz_gates(instruction.target_groups())
            elif instruction.name in cls._GATE_TO_EXPONENT:
                reducer._add_1_qubit_gates(
                    name=instruction.name,
                    targets=instruction.targets_copy(),
                )
            elif instruction.name != 'I':
                raise ValueError(f"Unsupported instruction {instruction.name} in circuit.")
        return out + reducer._to_circuit()

    def _to_circuit(self):
        """Materialize the currently accumulated gates as a circuit.

        :returns circuit: A ``stim.Circuit`` containing the gates accumulated in ``self``.
        """
        out = stim.Circuit()
        
        _1_qubit_gates: dict[int, set[int]] = defaultdict(set)
        for qubit_index, exponent in self.s_gates.items():
            _1_qubit_gates[exponent % self._PERIOD].add(qubit_index)
        for name, exponent in self._GATE_TO_EXPONENT.items():
            if targets := _1_qubit_gates[exponent]:
                out.append(name, targets, ())
        
        _max_moment = max(self.moment_to_cz.keys(), default=-1)
        for t in range(_max_moment + 1):
            for index_0, index_1 in self.moment_to_cz[t].items():
                if index_0 < index_1:
                    out.append("CZ", [index_0, index_1], ())
        
        return out

    def _add_1_qubit_gates(
            self,
            name: str,
            targets: list[stim.GateTarget],
    ):
        exponent = self._GATE_TO_EXPONENT[name]
        for target in targets:
            self.s_gates[target.value] += exponent

    def _add_cz_gates(self, target_groups: list[list[stim.GateTarget]]):
        for target_0, target_1 in target_groups:
            index_0, index_1 = target_0.value, target_1.value
            pair = frozenset({index_0, index_1})
            if pair in self.cz_to_moment.keys():
                t = self.cz_to_moment.pop(pair)
                self.moment_to_cz[t].pop(index_0)
                self.moment_to_cz[t].pop(index_1)
            else:
                t = self._get_earliest_moment(pair)
                self.cz_to_moment[pair] = t
                self.moment_to_cz[t][index_0] = index_1
                self.moment_to_cz[t][index_1] = index_0

    def _get_earliest_moment(self, pair: frozenset[int]):
        """Get the earliest timeslice when a specified CZ can be scheduled.

        :param pair: The unordered pair of physical qubit indices
            that the CZ gate should act upon.
        :returns moment: The earliest moment index that does not conflict.
        """
        t = 0
        while any(index in self.moment_to_cz[t] for index in pair):
            t += 1
        return t