from collections import Counter, defaultdict

import stim


class DepthReducer:

    _GATE_TO_EXPONENT = {'S': 1, 'Z': 2, 'S_DAG': 3}
    _PERIOD = 4

    def __init__(self) -> None:
        self.s_gates: Counter[int] = Counter()
        """A map from qubit index to the exponent of the S gate applied."""
        self.cz_to_moment: dict[frozenset[int], int] = {}
        """A map from unordered pair of qubit indices to the timeslice of its CZ gate."""
        self.moment_to_cz: defaultdict[int, dict[int, int]] = defaultdict(dict)
        """A map from timeslice to a bidirectional map from qubit index to qubit index."""
    
    @classmethod
    def reduce(cls, circuit: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        reducer = cls()
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                out += reducer._to_circuit()
                reducer = cls()
                out += instruction.repeat_count * cls.reduce(instruction.body_copy())
            elif instruction.name == 'H':
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
            else:
                raise ValueError(f"Unsupported instruction {instruction.name} in circuit.")
        return out + reducer._to_circuit()

    def _to_circuit(self):
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
        """Get earliest timeslice when CZ on `pair` can be scheduled."""
        t = 0
        while any(index in self.moment_to_cz[t] for index in pair):
            t += 1
        return t